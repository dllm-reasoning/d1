"""
Self-Distillation Trainer for DLLM

Training objective:
1. Forward pass 1 (Student): Prompt + fully masked response -> logits_uncond
2. Forward pass 2 (Teacher): Prompt + masked response with answer revealed -> logits_cond
3. Loss: KL divergence between logits_cond and logits_uncond on reasoning positions
"""

import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager

import torch
import torch.distributed as dist
import torch.nn.functional as F
from transformers import Trainer, DefaultDataCollator, TrainerCallback
import random
from tqdm import tqdm

# Import parser helpers from d1/eval/parser_helper.py — the same module used by
# parse_and_get_acc.py — so accuracy computation is identical to offline eval.
_eval_dir = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "eval"))
if _eval_dir not in sys.path:
    sys.path.insert(0, _eval_dir)
from parser_helper import remove_boxed, last_boxed_only_string, is_equiv as _ph_is_equiv


class SelfDistillationTrainer(Trainer):
    """
    Trainer for self-distillation with answer conditioning.
    """

    def __init__(
        self,
        *args,
        kl_weight=1.0,
        kl_type="forward",
        fixed_teacher=True,
        ema_teacher=False,
        ema_decay=0.999,
        loss_type="kl",
        ce_weight=1.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.kl_weight = kl_weight
        self.kl_type = kl_type  # "forward", "reverse", or "jsd"
        self.fixed_teacher = fixed_teacher  # If True, disable LoRA adapter for teacher (use base model)
        if ema_teacher and fixed_teacher:
            raise ValueError("ema_teacher and fixed_teacher are mutually exclusive — pick one.")
        self.ema_teacher = ema_teacher  # If True, use EMA of student weights as teacher
        self.ema_decay = ema_decay  # EMA decay rate (e.g. 0.999)
        self._ema_state: dict | None = None  # Lazily initialized on first training step
        self.loss_type = loss_type  # "kl", "kl_ce", or "ce"
        self.ce_weight = ce_weight  # Weight for CE loss on conditioning (answer) tokens

    # ------------------------------------------------------------------
    # EMA helpers
    # ------------------------------------------------------------------

    def _unwrap(self, model):
        return model.module if hasattr(model, "module") else model

    def _init_ema(self, model):
        """Initialize empty EMA shadow dict; params are populated lazily in _update_ema.

        With ZeRO-3, parameters may be size-0 shards at this point (before the
        first forward pass gathers them), so we cannot clone them here.
        """
        self._ema_state = {}

    def _update_ema(self, model):
        """EMA update: shadow = decay * shadow + (1 - decay) * param.

        Parameters are initialized lazily on first encounter with non-zero size,
        which is safe under ZeRO-3 because this runs after super().training_step()
        has already gathered params for the forward/backward pass.
        """
        unwrapped = self._unwrap(model)
        decay = self.ema_decay
        with torch.no_grad():
            for name, param in unwrapped.named_parameters():
                if not param.requires_grad:
                    continue
                p = param.detach().float()
                if p.numel() == 0:
                    # ZeRO-3: this rank holds no shard of this param right now; skip.
                    continue
                if name not in self._ema_state or self._ema_state[name].shape != p.shape:
                    # Lazy init (or re-init on shard-shape change).
                    self._ema_state[name] = p.clone()
                else:
                    self._ema_state[name].mul_(decay).add_(p, alpha=1.0 - decay)

    @contextmanager
    def _ema_teacher_context(self, model):
        """
        Temporarily swap trainable parameters to their EMA values for the
        teacher forward pass, then restore the originals.

        Memory cost: one extra copy of trainable params during the forward
        pass (negligible with LoRA; one full model copy with full fine-tuning).
        """
        unwrapped = self._unwrap(model)
        saved = {}
        for name, param in unwrapped.named_parameters():
            if param.requires_grad and name in self._ema_state:
                if param.data.shape != self._ema_state[name].shape:
                    # ZeRO-3: shard shape mismatch — skip swap for this param.
                    continue
                saved[name] = param.data  # save original storage (no copy needed)
                param.data = self._ema_state[name].to(dtype=param.dtype, device=param.device)
        try:
            yield
        finally:
            for name, param in unwrapped.named_parameters():
                if name in saved:
                    param.data = saved[name]  # restore original storage

    # ------------------------------------------------------------------
    # Override training_step to update EMA after each backward pass
    # ------------------------------------------------------------------

    def training_step(self, model, inputs, num_items_in_batch=None):
        # Initialize EMA on first step (model is on the right device by now)
        if self.ema_teacher and self._ema_state is None:
            self._init_ema(model)

        loss = super().training_step(model, inputs, num_items_in_batch)

        # Update EMA with the *pre-optimizer-step* weights.  The difference
        # vs. post-step is negligible over many steps.
        if self.ema_teacher and self._ema_state is not None:
            self._update_ema(model)

        return loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        with torch.no_grad():
            loss = self.compute_loss(model, inputs)
        return (loss.detach(), None, None)

    def evaluate(self, *args, **kwargs):
        print(f"\n{'='*60}")
        print(f"[Step {self.state.global_step}] Running KL LOSS evaluation on eval split...")
        print(f"{'='*60}")
        result = super().evaluate(*args, **kwargs)
        print(f"[Step {self.state.global_step}] KL loss eval done: {result}")
        print(f"{'='*60}\n")
        return result

    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
        """
        Self-distillation loss.

        loss_type="kl"    : KL divergence on reasoning (mask_positions) only — original behaviour.
        loss_type="kl_ce" : KL on reasoning + CE on teacher-conditioning answer span.
        loss_type="ce"    : CE on teacher-conditioning answer span only (no teacher forward pass).
        """
        # Extract inputs prepared by the data collator
        input_uncond = inputs["input_uncond"]  # All response masked
        input_cond = inputs["input_cond"]  # Answer revealed
        mask_positions = inputs["mask_positions"]  # Positions to compute KL loss (reasoning span)
        answer_mask = inputs["answer_mask"]  # Positions of GT conditioning tokens (answer span)
        # Front-reveal ablation gives cond/uncond different prompts, so they need separate
        # attention masks; the original end-reveal collator shares one mask for both.
        attention_mask_uncond = inputs.get("attention_mask_uncond", inputs.get("attention_mask"))
        attention_mask_cond = inputs.get("attention_mask_cond", inputs.get("attention_mask"))

        # Occasionally log what the teacher and student see for debugging.
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        if rank == 0 and self.model.training and random.random() < 0.01:
            tokenizer = self.data_collator.tokenizer
            mask_id = self.data_collator.mask_token_id

            def _decode(ids):
                return "".join(
                    (
                        "<MASK>"
                        if t.item() == mask_id
                        else tokenizer.decode([t.item()], skip_special_tokens=False)
                    )
                    for t in ids
                )

            print(f"\n{'='*60}")
            print(f"[Student (unconditional) @ step {self.state.global_step}]")
            print(_decode(input_uncond[0]))
            print(f"{'='*60}")
            print(f"[Teacher (conditional) @ step {self.state.global_step}]")
            print(_decode(input_cond[0]))
            print(f"{'='*60}\n")

        # Forward pass 1: Unconditional (student) — always needed
        outputs_uncond = model(input_uncond, attention_mask=attention_mask_uncond)
        logits_uncond = outputs_uncond.logits

        loss = torch.tensor(0.0, device=logits_uncond.device, dtype=logits_uncond.dtype)

        # ── KL loss on reasoning (mask_positions) ─────────────────────────────
        if self.loss_type in ("kl", "kl_ce"):
            # Forward pass 2: Conditional (teacher) - with no grad for teacher
            if self.ema_teacher:
                if self._ema_state is not None:
                    with self._ema_teacher_context(model):
                        with torch.no_grad():
                            outputs_cond = model(input_cond, attention_mask=attention_mask_cond)
                else:
                    with torch.no_grad():
                        outputs_cond = model(input_cond, attention_mask=attention_mask_cond)
            elif self.fixed_teacher:
                unwrapped_model = self._unwrap(model)
                with torch.no_grad():
                    unwrapped_model.disable_adapter_layers()
                    outputs_cond = model(input_cond, attention_mask=attention_mask_cond)
                    unwrapped_model.enable_adapter_layers()
            else:
                with torch.no_grad():
                    outputs_cond = model(input_cond, attention_mask=attention_mask_cond)
            logits_cond = outputs_cond.logits

            log_p_cond = F.log_softmax(logits_cond, dim=-1)
            log_p_uncond = F.log_softmax(logits_uncond, dim=-1)
            p_cond = F.softmax(logits_cond, dim=-1)
            p_uncond = F.softmax(logits_uncond, dim=-1)

            if self.kl_type == "forward":
                kl_div = (p_cond * (log_p_cond - log_p_uncond)).sum(dim=-1)
            elif self.kl_type == "reverse":
                kl_div = (p_uncond * (log_p_uncond - log_p_cond)).sum(dim=-1)
            else:  # jsd
                m = 0.5 * (p_cond + p_uncond)
                log_m = m.log()
                kl_div = 0.5 * (p_cond * (log_p_cond - log_m) + p_uncond * (log_p_uncond - log_m)).sum(dim=-1)

            kl_div = kl_div * mask_positions.float()
            kl_loss = kl_div.sum() / (mask_positions.sum() + 1e-8)
            loss = loss + self.kl_weight * kl_loss
        else:
            kl_loss = torch.tensor(0.0)

        # ── CE loss on teacher-conditioning answer span ────────────────────────
        if self.loss_type in ("kl_ce", "ce"):
            # Student logits at answer positions; GT tokens from input_cond (answer revealed there)
            ce_logits = logits_uncond[answer_mask]   # (n_ans, V)
            ce_targets = input_cond[answer_mask]     # (n_ans,)
            ce_loss = F.cross_entropy(ce_logits, ce_targets)
            loss = loss + self.ce_weight * ce_loss
        else:
            ce_loss = torch.tensor(0.0)

        # ── Logging ───────────────────────────────────────────────────────────
        log_dict = {}
        if self.loss_type in ("kl", "kl_ce"):
            log_dict["kl_loss" if self.model.training else "eval_kl_loss"] = kl_loss.item()
        if self.loss_type in ("kl_ce", "ce"):
            log_dict["ce_loss" if self.model.training else "eval_ce_loss"] = ce_loss.item()

        if self.model.training:
            if (self.state.global_step + 1) % self.args.logging_steps == 0:
                self.log(log_dict)
        else:
            self.log(log_dict)

        return loss if not return_outputs else (loss, outputs_uncond)


class SelfDistillationDataCollator(DefaultDataCollator):
    """
    Data collator for self-distillation training.
    Prepares both unconditional and conditional masked inputs.
    """

    def __init__(self, tokenizer, mask_token_id=126336, **kwargs):
        super().__init__()
        self.tokenizer = tokenizer
        self.mask_token_id = mask_token_id
        self.pad_token_id = tokenizer.pad_token_id
        if tokenizer.mask_token_id is not None:
            self.mask_token_id = tokenizer.mask_token_id

    def __call__(self, batch):
        # All sequences have the same fixed length (MAX_PROMPT_LENGTH + MAX_COMPLETION_LENGTH)
        # after prompt right-padding in preprocessing — no dynamic padding needed.
        input_ids = torch.stack([item["input_ids"] for item in batch])
        prompt_lengths = torch.tensor([item["prompt_lengths"] for item in batch])
        answer_starts = torch.tensor([item["answer_start"] for item in batch])
        answer_ends = torch.tensor([item["answer_end"] for item in batch])

        B, L = input_ids.shape

        # Create unconditional input: mask from end-of-prompt to end-of-answer
        input_uncond = input_ids.clone()
        for i in range(B):
            input_uncond[i, prompt_lengths[i] : answer_ends[i]] = self.mask_token_id

        # Create conditional input: same masking but reveal the answer span
        input_cond = input_ids.clone()
        for i in range(B):
            input_cond[i, prompt_lengths[i] : answer_ends[i]] = self.mask_token_id
            input_cond[i, answer_starts[i] : answer_ends[i]] = input_ids[i, answer_starts[i] : answer_ends[i]]

        # Create mask_positions: reasoning span only (for KL loss)
        mask_positions = torch.zeros(B, L, dtype=torch.bool)
        for i in range(B):
            mask_positions[i, prompt_lengths[i] : answer_starts[i]] = True

        # Create answer_mask: answer span (for CE loss on conditioning tokens)
        answer_mask = torch.zeros(B, L, dtype=torch.bool)
        for i in range(B):
            answer_mask[i, answer_starts[i] : answer_ends[i]] = True

        # Attention mask: 1 for real tokens (prompt + reasoning + answer), 0 for padding
        # prompt-pad and completion-pad both use pad_token_id; mask tokens (126336) are not pad_token_id
        attention_mask = (input_ids != self.pad_token_id).long()

        return {
            "input_uncond": input_uncond,
            "input_cond": input_cond,
            "mask_positions": mask_positions,
            "answer_mask": answer_mask,
            "attention_mask": attention_mask,
        }


class SelfDistillationDataCollatorFrontReveal(DefaultDataCollator):
    """
    Data collator for the "privileged info in the front" ablation.

    Unlike SelfDistillationDataCollator, cond and uncond carry different prompts
    (built upstream by _preprocess_dual_prompt_masked_completion) and the
    completion is fully masked for both — there's nothing to reveal inside the
    completion since the answer is already given away in the cond prompt. Since
    the prompts differ in real-token length, cond and uncond need their own
    attention masks (unlike the shared one in SelfDistillationDataCollator).
    """

    def __init__(self, tokenizer, mask_token_id=126336, **kwargs):
        super().__init__()
        self.tokenizer = tokenizer
        self.mask_token_id = mask_token_id
        self.pad_token_id = tokenizer.pad_token_id
        if tokenizer.mask_token_id is not None:
            self.mask_token_id = tokenizer.mask_token_id

    def __call__(self, batch):
        input_uncond = torch.stack([item["input_ids_uncond"] for item in batch])
        input_cond = torch.stack([item["input_ids_cond"] for item in batch])
        completion_starts = torch.tensor([item["completion_start"] for item in batch])
        completion_lens = torch.tensor([item["completion_len"] for item in batch])

        B, L = input_uncond.shape

        # mask_positions is identical for cond/uncond here: completion_start is the
        # same fixed value (max_prompt_length) for every row by construction.
        mask_positions = torch.zeros(B, L, dtype=torch.bool)
        for i in range(B):
            mask_positions[i, completion_starts[i] : completion_starts[i] + completion_lens[i]] = True

        # No revealed answer span in this ablation — CE-on-answer-span loss doesn't apply.
        answer_mask = torch.zeros(B, L, dtype=torch.bool)

        attention_mask_uncond = (input_uncond != self.pad_token_id).long()
        attention_mask_cond = (input_cond != self.pad_token_id).long()

        return {
            "input_uncond": input_uncond,
            "input_cond": input_cond,
            "mask_positions": mask_positions,
            "answer_mask": answer_mask,
            "attention_mask_uncond": attention_mask_uncond,
            "attention_mask_cond": attention_mask_cond,
        }


class SelfDistillationDataset(torch.utils.data.Dataset):
    """
    Dataset for self-distillation training.
    """

    def __init__(self, data, tokenizer, max_length):
        super().__init__()
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


SYSTEM_PROMPT = """You are a math expert. You will be given a question to solve. Solve it step by step. Wrap the final answer in a \\boxed{}. 
Respond in the following format:
<reasoning>
Your reasoning here
</reasoning>
<answer>
\\boxed{...}
</answer>"""


MCQ_SYSTEM_PROMPT = """Respond in the following format:
<reasoning>
Your reasoning here
</reasoning>
<answer>
\\boxed{X}
</answer>
where X is one of A, B, C, or D. Wrap the letter in \\boxed{{}} to indicate your final answer choice."""

MBPP_SYSTEM_PROMPT = """You are a coding expert. You will be given a coding problem to solve. Solve it step by step. Ensure you wrap the answer in ```python```.

Respond in the following format:
<reasoning>
Your reasoning here
</reasoning>
<answer>
```python
Your code here
```
</answer>"""

HUMANEVAL_SYSTEM_PROMPT = """You are a coding expert. You will be given a coding problem to solve. Solve it step by step.

Respond in the following format:
<reasoning>
Your reasoning here
</reasoning>
<answer>
```python
Your code here
```
</answer>"""


def _preprocess_masked_reasoning(
    items,
    tokenizer,
    max_length,
    min_reasoning_tokens,
    max_reasoning_tokens,
    desc,
    max_prompt_length=None,
    cache_dir=None,
):
    """
    Shared MASK-injection preprocessing for self-distillation.

    No real reasoning trace is used. The reasoning span is filled with MASK
    tokens of a randomly sampled length so the model must learn to infer it
    conditioned on the revealed answer span.

    input_ids layout: [prompt_tokens][mask * reasoning_len][answer_tokens][pad]

    Args:
        items: list of {"question_text": str, "answer_text": str}
               answer_text must include the </reasoning> boundary, e.g.:
               '</reasoning>\\n<answer>\\boxed{42}</answer>'
        max_prompt_length: if set, truncates the prompt to this many tokens. Used only
                           for prompt truncation — does not affect the completion span.
    """
    mask_token_id = tokenizer.mask_token_id if tokenizer.mask_token_id is not None else 126336
    # max_length is the completion span length (not total sequence length).
    # max_prompt_length is only used for prompt truncation.
    prompt_trunc_len = max_prompt_length if max_prompt_length is not None else max_length

    if cache_dir is not None:
        cache_key = hashlib.md5(
            f"{desc}|{max_length}|{min_reasoning_tokens}|{max_reasoning_tokens}|{prompt_trunc_len}|{len(items)}".encode()
        ).hexdigest()[:16]
        cache_file = os.path.join(cache_dir, f"preprocess_{cache_key}.pt")
        if os.path.exists(cache_file):
            print(f"[Cache] Loading preprocessed data from {cache_file}")
            return torch.load(cache_file, weights_only=False)

    preprocessed_data = []

    for i, item in enumerate(tqdm(items, desc=desc)):
        question_text = item["question_text"]
        answer_text = item["answer_text"]

        # Tokenize prompt (truncated independently)
        prompt_msgs = [{"role": "user", "content": question_text}]
        prompt_text = tokenizer.apply_chat_template(prompt_msgs, tokenize=False) + "\n"
        tokenized_prompt = tokenizer(
            prompt_text, return_tensors="pt", truncation=True, max_length=prompt_trunc_len
        )
        prompt_ids = tokenized_prompt.input_ids.squeeze(0)
        prompt_length = tokenized_prompt.attention_mask.sum().item()

        # Tokenize answer span
        answer_ids = tokenizer(answer_text, add_special_tokens=False, return_tensors="pt").input_ids.squeeze(
            0
        )
        answer_len = len(answer_ids)

        # Completion span is exactly max_length tokens: [masked_reasoning | answer | pad]
        max_reasoning_len = max_length - answer_len
        if max_reasoning_len <= 0:
            continue  # Answer alone overflows completion span
        reasoning_len = min(random.randint(min_reasoning_tokens, max_reasoning_tokens), max_reasoning_len)

        # Build input_ids: [prompt][mask * reasoning_len][answer][pad]
        # Padding fills to prompt_trunc_len + max_length so all sequences are the same total length.
        padding_len = prompt_trunc_len + max_length - prompt_length - reasoning_len - answer_len
        reasoning_ids = torch.full((reasoning_len,), mask_token_id, dtype=torch.long)
        pad_ids = torch.full((padding_len,), tokenizer.pad_token_id, dtype=torch.long)
        input_ids = torch.cat([prompt_ids[:prompt_length], reasoning_ids, answer_ids, pad_ids])

        if i == 0:
            print(f"\n{'='*60}")
            print(f"[{desc}] Sample (i=0):")
            print(f"  answer_text   : {answer_text!r}")
            print(
                f"  prompt_length : {prompt_length}, reasoning_len: {reasoning_len}, answer_len: {answer_len}, padding_len: {padding_len}"
            )
            print(
                f"  answer_start  : {prompt_length + reasoning_len}, answer_end: {prompt_length + reasoning_len + answer_len}"
            )
            print(f"{'='*60}\n")

        preprocessed_data.append(
            {
                "input_ids": input_ids,
                "prompt_lengths": prompt_length,
                "answer_start": prompt_length + reasoning_len,
                "answer_end": prompt_length + reasoning_len + answer_len,
            }
        )

    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        torch.save(preprocessed_data, cache_file)
        print(f"[Cache] Saved preprocessed data to {cache_file}")

    return preprocessed_data


def _preprocess_dual_prompt_masked_completion(
    items,
    tokenizer,
    max_length,
    min_reasoning_tokens,
    max_reasoning_tokens,
    desc,
    max_prompt_length,
    cache_dir=None,
):
    """
    Preprocessing for the "privileged info in the front" ablation.

    Unlike _preprocess_masked_reasoning (where cond/uncond share one prompt and
    only the trailing answer span differs), here cond and uncond get genuinely
    different prompts:
      - question_text_uncond: no privileged info
      - question_text_cond:   question + a sentence revealing the ground-truth answer

    The completion (reasoning + answer span) is fully masked for BOTH branches —
    there's nothing left to reveal inside the completion since the answer is
    already given away in the cond prompt.

    Both prompts are LEFT-padded to the same fixed `max_prompt_length` slot so
    the completion starts at the same absolute position for cond and uncond —
    required for the per-position KL loss in compute_loss to line up. Left
    padding (not right) keeps prompt and completion contiguous: pads between
    prompt and completion are attention-masked out but still consume RoPE
    positions, which pushes the completion into "EOS padding territory" and
    collapses its predictions toward <|endoftext|> (verified empirically).
    This matches the eval callbacks' _left_pad_batch convention.

    input_ids layout (both branches): [pad * (max_prompt_length - prompt_len)][prompt][MASK * (reasoning_len+answer_len)][pad]
    """
    mask_token_id = tokenizer.mask_token_id if tokenizer.mask_token_id is not None else 126336

    if cache_dir is not None:
        cache_key = hashlib.md5(
            f"{desc}|front|{max_length}|{min_reasoning_tokens}|{max_reasoning_tokens}|{max_prompt_length}|{len(items)}".encode()
        ).hexdigest()[:16]
        cache_file = os.path.join(cache_dir, f"preprocess_{cache_key}.pt")
        if os.path.exists(cache_file):
            print(f"[Cache] Loading preprocessed data from {cache_file}")
            return torch.load(cache_file, weights_only=False)

    def _tokenize_prompt(text):
        prompt_text = tokenizer.apply_chat_template([{"role": "user", "content": text}], tokenize=False) + "\n"
        tok = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=max_prompt_length)
        length = tok.attention_mask.sum().item()
        return tok.input_ids.squeeze(0)[:length]

    preprocessed_data = []
    n_skipped_overflow = 0

    for i, item in enumerate(tqdm(items, desc=desc)):
        uncond_ids = _tokenize_prompt(item["question_text_uncond"])
        cond_ids = _tokenize_prompt(item["question_text_cond"])

        if len(cond_ids) >= max_prompt_length:
            # Reference-solution sentence got truncated away by max_prompt_length — skip,
            # the ablation requires the privileged-info sentence to survive intact.
            n_skipped_overflow += 1
            continue

        answer_ids = tokenizer(
            item["answer_text"], add_special_tokens=False, return_tensors="pt"
        ).input_ids.squeeze(0)
        answer_len = len(answer_ids)
        max_reasoning_len = max_length - answer_len
        if max_reasoning_len <= 0:
            continue
        reasoning_len = min(random.randint(min_reasoning_tokens, max_reasoning_tokens), max_reasoning_len)
        completion_len = reasoning_len + answer_len

        mask_ids = torch.full((completion_len,), mask_token_id, dtype=torch.long)
        tail_pad_ids = torch.full((max_length - completion_len,), tokenizer.pad_token_id, dtype=torch.long)

        def _build(prompt_ids):
            prompt_pad_ids = torch.full(
                (max_prompt_length - len(prompt_ids),), tokenizer.pad_token_id, dtype=torch.long
            )
            # Left-pad the prompt so it stays contiguous with the completion (see docstring).
            return torch.cat([prompt_pad_ids, prompt_ids, mask_ids, tail_pad_ids])

        input_ids_uncond = _build(uncond_ids)
        input_ids_cond = _build(cond_ids)

        if i == 0:
            print(f"\n{'='*60}")
            print(f"[{desc}] Sample (i=0, front-reveal):")
            print(f"  uncond prompt_len: {len(uncond_ids)}, cond prompt_len: {len(cond_ids)}")
            print(f"  reasoning_len: {reasoning_len}, answer_len: {answer_len}")
            print(f"  completion_start (fixed for both branches): {max_prompt_length}")
            print(f"{'='*60}\n")

        preprocessed_data.append(
            {
                "input_ids_uncond": input_ids_uncond,
                "input_ids_cond": input_ids_cond,
                "completion_start": max_prompt_length,
                "completion_len": completion_len,
            }
        )

    if n_skipped_overflow:
        print(
            f"[{desc}] Skipped {n_skipped_overflow}/{len(items)} examples where the "
            f"reference-solution sentence overflowed max_prompt_length={max_prompt_length}"
        )

    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        torch.save(preprocessed_data, cache_file)
        print(f"[Cache] Saved preprocessed data to {cache_file}")

    return preprocessed_data


def preprocess_sciknoweval_for_self_distillation(
    data,
    tokenizer,
    max_length,
    min_reasoning_tokens=128,
    max_reasoning_tokens=256,
    max_prompt_length=None,
    privileged_info_position="end",
):
    """
    privileged_info_position:
      "end"   (default) — answer revealed as a trailing completion span (original SFCD).
      "front" — answer revealed as a sentence appended to the prompt (OPSD-style);
                the completion (reasoning + answer) is fully masked for both branches.
    """
    items = []
    for i in range(len(data)):
        item = data[i]
        choices = item["choices"]
        formatted_choices = "\n".join(
            f"{label}. {text}" for label, text in zip(choices["label"], choices["text"])
        )
        base_question = MCQ_SYSTEM_PROMPT + "\n\n" + item["question"] + "\n\n" + formatted_choices
        answer_key = item["answerKey"]  # "A", "B", "C", or "D"
        answer_text = f"</reasoning>\n<answer>\\boxed{{{answer_key}}}</answer>"

        if privileged_info_position == "front":
            reference_sentence = (
                f"\n\nHere is the reference solution: {answer_key}. Understand it and then "
                "generate the reasoning steps and answer in the format above."
            )
            items.append(
                {
                    "question_text_uncond": base_question,
                    "question_text_cond": base_question + reference_sentence,
                    "answer_text": answer_text,
                }
            )
        else:
            items.append({"question_text": base_question, "answer_text": answer_text})

    if privileged_info_position == "front":
        return _preprocess_dual_prompt_masked_completion(
            items,
            tokenizer,
            max_length,
            min_reasoning_tokens,
            max_reasoning_tokens,
            desc="Preprocessing SciKnowEval (front-reveal)",
            max_prompt_length=max_prompt_length or max_length,
        )

    return _preprocess_masked_reasoning(
        items,
        tokenizer,
        max_length,
        min_reasoning_tokens,
        max_reasoning_tokens,
        desc="Preprocessing SciKnowEval",
        max_prompt_length=max_prompt_length,
    )


def preprocess_gsm8k_for_self_distillation(
    data,
    tokenizer,
    max_length,
    min_reasoning_tokens=128,
    max_reasoning_tokens=256,
    test_split=0.01,
):
    items = []
    for i in range(len(data)):
        question_text = SYSTEM_PROMPT + "\n\n" + data[i]["question"]
        answer_parts = data[i]["answer"].split("####")
        final_answer = answer_parts[1].strip() if len(answer_parts) > 1 else ""
        answer_text = f"</reasoning>\n<answer>\\boxed{{{final_answer}}}</answer>"
        items.append({"question_text": question_text, "answer_text": answer_text})

    random.shuffle(items)
    split = int(len(items) * test_split)
    train_items, test_items = items[split:], items[:split]

    train_data = _preprocess_masked_reasoning(
        train_items,
        tokenizer,
        max_length,
        min_reasoning_tokens,
        max_reasoning_tokens,
        desc="Preprocessing GSM8K train",
    )
    test_data = _preprocess_masked_reasoning(
        test_items,
        tokenizer,
        max_length,
        min_reasoning_tokens,
        max_reasoning_tokens,
        desc="Preprocessing GSM8K eval",
    )
    return train_data, test_data


def preprocess_kodcode_for_self_distillation(
    data,
    tokenizer,
    max_length,
    min_reasoning_tokens=128,
    max_reasoning_tokens=512,
    test_split=0.01,
    max_prompt_length=None,
):
    items = []
    for i in range(len(data)):
        question_text = MBPP_SYSTEM_PROMPT + "\n\n" + data[i]["question"]
        solution = data[i]["solution"]
        answer_text = f"</reasoning>\n<answer>\n```python\n{solution}\n```\n</answer>"
        items.append({"question_text": question_text, "answer_text": answer_text})

    random.shuffle(items)
    split = int(len(items) * test_split)
    train_items, test_items = items[split:], items[:split]

    train_data = _preprocess_masked_reasoning(
        train_items,
        tokenizer,
        max_length,
        min_reasoning_tokens,
        max_reasoning_tokens,
        desc="Preprocessing KodCode train",
        max_prompt_length=max_prompt_length,
    )
    test_data = _preprocess_masked_reasoning(
        test_items,
        tokenizer,
        max_length,
        min_reasoning_tokens,
        max_reasoning_tokens,
        desc="Preprocessing KodCode eval",
        max_prompt_length=max_prompt_length,
    )
    return train_data, test_data


def build_mbpp_eval_prompts(tokenizer, n=50, max_prompt_length=512, seed=42):
    """
    Sample n examples from MBPP sanitized test set and return:
        {"prompt_ids": Tensor[L], "test_code": str}
    where test_code = test_imports + test_list (matching mbpp.py's answer field).
    """
    from datasets import load_dataset as hf_load_dataset

    data = hf_load_dataset("mbpp", "sanitized", split="test")
    indices = random.Random(seed).sample(range(len(data)), min(n, len(data)))
    eval_items = []
    for i in indices:
        item = data[i]
        unit_tests = "\n".join(item["test_list"])
        question_text = (
            MBPP_SYSTEM_PROMPT
            + "\n\n"
            + item["prompt"]
            + "\n\nYour code should pass the following tests:\n\n"
            + unit_tests
        )
        test_imports = "\n".join(item["test_imports"])
        test_code = (test_imports + "\n\n" + unit_tests).strip()
        prompt_msgs = [{"role": "user", "content": question_text}]
        prompt_text = tokenizer.apply_chat_template(prompt_msgs, tokenize=False) + "\n"
        prompt_ids = tokenizer(
            prompt_text, return_tensors="pt", truncation=True, max_length=max_prompt_length
        ).input_ids.squeeze(0)
        eval_items.append({"prompt_ids": prompt_ids, "test_code": test_code})
    return eval_items


def _get_code_eval_helpers():
    """Load Parser, test_solution, extract_human_eval_prompt from the shared eval directory."""
    eval_dir = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "eval"))
    if eval_dir not in sys.path:
        sys.path.insert(0, eval_dir)
    from parsers import Parser, test_solution, extract_human_eval_prompt  # noqa: PLC0415

    return Parser, test_solution, extract_human_eval_prompt


def build_humaneval_eval_prompts(tokenizer, n=50, max_prompt_length=512, seed=42):
    """
    Sample n examples from HumanEval test set and return:
        {"prompt_ids": Tensor[L], "test": str}
    where test is item["test"] from the dataset (matching human_eval.py's answer field).
    """
    from datasets import load_dataset as hf_load_dataset

    _, _, extract_human_eval_prompt = _get_code_eval_helpers()
    data = hf_load_dataset("openai/openai_humaneval", split="test")
    indices = random.Random(seed).sample(range(len(data)), min(n, len(data)))
    eval_items = []
    for i in indices:
        item = data[i]
        question_text = HUMANEVAL_SYSTEM_PROMPT + "\n\n" + extract_human_eval_prompt(item["prompt"])
        prompt_msgs = [{"role": "user", "content": question_text}]
        prompt_text = tokenizer.apply_chat_template(prompt_msgs, tokenize=False) + "\n"
        prompt_ids = tokenizer(
            prompt_text, return_tensors="pt", truncation=True, max_length=max_prompt_length
        ).input_ids.squeeze(0)
        eval_items.append({"prompt_ids": prompt_ids, "test": item["test"]})
    return eval_items


# ---------------------------------------------------------------------------
# SciKnowEval in-training evaluation
# ---------------------------------------------------------------------------


def _safe_rank():
    import torch.distributed as dist

    return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0


def _extract_last_boxed_content(text):
    """Return content of the last \\boxed{...} (or \\fbox{...}) expression."""
    if text is None:
        return None
    start = max(text.rfind("\\boxed"), text.rfind("\\fbox"))
    if start < 0:
        return None

    brace_start = text.find("{", start)
    if brace_start < 0:
        return None

    depth = 0
    for i in range(brace_start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[brace_start + 1 : i].strip()
    return None


def _math_answers_equal(pred, gold):
    """
    Compare math answers with parser-based normalization when available.
    Falls back to numeric/string equality.
    """
    if pred is None or gold is None:
        return False

    pred_norm = pred.replace(",", "").strip()
    gold_norm = gold.replace(",", "").strip()

    # Try numeric comparison first so that e.g. "7.00" == "7" is handled correctly.
    try:
        if abs(float(pred_norm) - float(gold_norm)) < 1e-6:
            return True
    except Exception:
        pass

    eval_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "eval")
    if eval_dir not in sys.path:
        sys.path.insert(0, eval_dir)
    try:
        from parsers import is_equiv  # noqa: PLC0415

        return bool(is_equiv(pred, gold))
    except Exception:
        pass

    return pred_norm == gold_norm


def _parse_mcq_answer(text):
    """Extract MCQ answer from last boxed expression."""
    boxed = _extract_last_boxed_content(text)
    if boxed is None:
        return None
    letter = boxed.strip().upper()
    return letter if letter in {"A", "B", "C", "D"} else None


def _parse_gsm8k_gold_answer(answer_text):
    parts = answer_text.split("####")
    return parts[-1].strip() if parts else answer_text.strip()


def _parse_gsm8k_pred_answer(text):
    return _extract_last_boxed_content(text)


def build_sciknoweval_eval_prompts(data, tokenizer, n=50, max_prompt_length=512):
    """
    Sample n examples from a raw SciKnowEval HF dataset and return a list of
        {"prompt_ids": Tensor[L], "answer_key": str}
    ready for the SciKnowEvalCallback.
    """
    indices = random.sample(range(len(data)), min(n, len(data)))
    eval_items = []
    for i in indices:
        item = data[i]
        choices = item["choices"]
        formatted_choices = "\n".join(
            f"{label}. {text}" for label, text in zip(choices["label"], choices["text"])
        )
        question_text = MCQ_SYSTEM_PROMPT + "\n\n" + item["question"] + "\n\n" + formatted_choices
        prompt_msgs = [{"role": "user", "content": question_text}]
        prompt_text = tokenizer.apply_chat_template(prompt_msgs, tokenize=False) + "\n"
        prompt_ids = tokenizer(
            prompt_text, return_tensors="pt", truncation=True, max_length=max_prompt_length
        ).input_ids.squeeze(0)
        eval_items.append({"prompt_ids": prompt_ids, "answer_key": item["answerKey"]})
    return eval_items


def build_math500_eval_prompts(tokenizer, n=500, max_prompt_length=512, seed=42):
    """
    Sample n examples from HuggingFaceH4/MATH-500 test set and return:
        {"prompt_ids": Tensor[L], "answer_key": str}
    where answer_key is the expected final answer (already extracted, e.g. "\\frac{1}{2}").
    """
    from datasets import load_dataset as hf_load_dataset

    data = hf_load_dataset("HuggingFaceH4/MATH-500", split="test")
    indices = random.Random(seed).sample(range(len(data)), min(n, len(data)))
    eval_items = []
    for i in indices:
        item = data[i]
        question_text = SYSTEM_PROMPT + "\n\n" + item["problem"]
        answer_key = item["answer"]  # already extracted, e.g. "\frac{1}{2}"
        prompt_msgs = [{"role": "user", "content": question_text}]
        prompt_text = (
            tokenizer.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
            + "<reasoning>"
        )
        prompt_ids = tokenizer(
            prompt_text, return_tensors="pt", truncation=True, max_length=max_prompt_length
        ).input_ids.squeeze(0)
        eval_items.append({"prompt_ids": prompt_ids, "answer_key": answer_key})
    return eval_items


def preprocess_math500_for_self_distillation(
    data,
    tokenizer,
    max_length,
    min_reasoning_tokens=128,
    max_reasoning_tokens=256,
    test_split=0.01,
):
    """Preprocess ankner/math-500 (train split) for self-distillation.

    The dataset's `solution` field is the full solution text; we extract the
    last \\boxed{} content from it to get the clean final answer.
    """
    items = []
    for i in range(len(data)):
        question_text = SYSTEM_PROMPT + "\n\n" + data[i]["problem"]
        final_answer = _extract_last_boxed_content(data[i]["solution"])
        if final_answer is None:
            continue  # skip examples where we can't parse an answer
        answer_text = f"</reasoning>\n<answer>\\boxed{{{final_answer}}}</answer>"
        items.append({"question_text": question_text, "answer_text": answer_text})

    random.shuffle(items)
    split = int(len(items) * test_split)
    train_items, test_items = items[split:], items[:split]

    train_data = _preprocess_masked_reasoning(
        train_items,
        tokenizer,
        max_length,
        min_reasoning_tokens,
        max_reasoning_tokens,
        desc="Preprocessing MATH train",
    )
    test_data = _preprocess_masked_reasoning(
        test_items,
        tokenizer,
        max_length,
        min_reasoning_tokens,
        max_reasoning_tokens,
        desc="Preprocessing MATH eval",
    )
    return train_data, test_data


def build_math500_eval_prompts(tokenizer, n=500, max_prompt_length=512, seed=42):
    """
    Sample n examples from HuggingFaceH4/MATH-500 test set and return:
        {"prompt_ids": Tensor[L], "answer_key": str}
    where answer_key is the extracted final answer (e.g. "\\frac{1}{2}").
    """
    from datasets import load_dataset as hf_load_dataset

    data = hf_load_dataset("HuggingFaceH4/MATH-500", split="test")
    indices = random.Random(seed).sample(range(len(data)), min(n, len(data)))
    eval_items = []
    for i in indices:
        item = data[i]
        question_text = SYSTEM_PROMPT + "\n\n" + item["problem"]
        answer_key = item["answer"]  # already extracted, e.g. "\frac{1}{2}"
        prompt_msgs = [{"role": "user", "content": question_text}]
        prompt_text = (
            tokenizer.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
            + "<reasoning>"
        )
        prompt_ids = tokenizer(
            prompt_text, return_tensors="pt", truncation=True, max_length=max_prompt_length
        ).input_ids.squeeze(0)
        eval_items.append({"prompt_ids": prompt_ids, "answer_key": answer_key})
    return eval_items


def build_gsm8k_eval_prompts(data, tokenizer, n=50, max_prompt_length=512, seed=42):
    """
    Sample n examples from raw GSM8K data and return:
        {"prompt_ids": Tensor[L], "answer_key": str}
    where answer_key is the expected final numeric answer.
    """
    # Use a dedicated RNG so callback eval set is stable across runs.
    indices = random.Random(seed).sample(range(len(data)), min(n, len(data)))
    eval_items = []
    for i in indices:
        item = data[i]
        question_text = SYSTEM_PROMPT + "\n\n" + item["question"]
        answer_key = _parse_gsm8k_gold_answer(item["answer"])
        prompt_msgs = [{"role": "user", "content": question_text}]
        prompt_text = tokenizer.apply_chat_template(prompt_msgs, tokenize=False) + "\n"
        prompt_ids = tokenizer(
            prompt_text, return_tensors="pt", truncation=True, max_length=max_prompt_length
        ).input_ids.squeeze(0)
        eval_items.append({"prompt_ids": prompt_ids, "answer_key": answer_key})
    return eval_items


_generate_fn = None


def _get_generate():
    """Load generate() from the shared eval directory."""
    global _generate_fn
    if _generate_fn is None:
        import importlib.util

        gen_path = os.path.normpath(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "eval", "generate.py")
        )
        spec = importlib.util.spec_from_file_location("sft_eval_generate", gen_path)
        gen_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(gen_mod)
        _generate_fn = gen_mod.generate
    return _generate_fn


def _dllm_generate(model, prompt, tokenizer, attention_mask=None, **kwargs):
    """
    Thin wrapper around eval/generate.py's `generate` that is safe in both
    distributed and single-GPU contexts.  (generate.py calls dist.get_rank()
    unconditionally; we monkeypatch it when dist is not initialised.)
    """
    import torch.distributed as _dist

    generate = _get_generate()

    if not (_dist.is_available() and _dist.is_initialized()):
        _orig = _dist.get_rank
        _dist.get_rank = lambda: 0
        try:
            return generate(model, prompt, attention_mask, **kwargs)
        finally:
            _dist.get_rank = _orig
    return generate(model, prompt, attention_mask, **kwargs)


class TrainingTimeCallback(TrainerCallback):
    """Tracks cumulative wall-clock training time (excluding eval) and logs it to wandb.

    Register this callback *before* any eval callbacks so that its on_step_end
    fires immediately after the optimizer step, before eval runs.
    """

    def __init__(self):
        self._step_start: float | None = None
        self._total_train_time_s: float = 0.0

    def on_step_begin(self, args, state, control, **kwargs):
        self._step_start = time.perf_counter()

    def on_step_end(self, args, state, control, **kwargs):
        if self._step_start is not None:
            self._total_train_time_s += time.perf_counter() - self._step_start
            self._step_start = None
        try:
            import wandb

            if wandb.run is not None:
                wandb.log({"train/training_time_s": self._total_train_time_s})
        except Exception:
            pass


class SciKnowEvalCallback(TrainerCallback):
    """
    Runs MCQ accuracy evaluation on a fixed 50-example SciKnowEval split
    every `eval_steps` training steps and logs accuracy to wandb.

    gen_length must be divisible by block_length (default 32).
    """

    def __init__(
        self,
        eval_items,
        tokenizer,
        gen_length=256,
        block_length=32,
        eval_steps=50,
        eval_batch_size=4,
        output_dir=None,
        diffusion_steps=None,
        tokens_per_step=None,
        mask_id=126336,
        skip_first_step_eval=False,
    ):
        assert gen_length % block_length == 0, "gen_length must be divisible by block_length"
        self.eval_items = eval_items
        self.tokenizer = tokenizer
        self.gen_length = gen_length
        self.block_length = block_length
        self.eval_steps = eval_steps
        self.eval_batch_size = eval_batch_size
        self.output_dir = output_dir
        if diffusion_steps is not None:
            self.diffusion_steps = diffusion_steps
        elif tokens_per_step is not None:
            self.diffusion_steps = max(1, gen_length // tokens_per_step)
        else:
            self.diffusion_steps = gen_length
        self.mask_id = mask_id
        self.skip_first_step_eval = skip_first_step_eval

    def _left_pad_batch(self, prompt_ids_list, device):
        """Left-pad a list of 1D prompt tensors to the same length and return (padded, attention_mask)."""
        max_len = max(p.shape[0] for p in prompt_ids_list)
        pad_id = self.tokenizer.pad_token_id
        padded = torch.full((len(prompt_ids_list), max_len), pad_id, dtype=torch.long, device=device)
        attention_mask = torch.zeros(len(prompt_ids_list), max_len, dtype=torch.long, device=device)
        for i, p in enumerate(prompt_ids_list):
            padded[i, max_len - p.shape[0] :] = p.to(device)
            attention_mask[i, max_len - p.shape[0] :] = 1
        return padded, attention_mask

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        """Evaluate the base model before any training steps."""
        if self.skip_first_step_eval:
            return
        self._run_eval(args, state, model)

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step % self.eval_steps != 0:
            return
        self._run_eval(args, state, model)

    def _run_eval(self, args, state, model):
        import torch.distributed as dist

        rank = _safe_rank()
        world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1

        device = next(model.parameters()).device
        was_training = model.training
        model.eval()

        if rank == 0:
            print(f"\n{'='*60}")
            print(
                f"[Step {state.global_step}] Running ACCURACY evaluation on {len(self.eval_items)} SciKnowEval questions..."
            )
            print(
                f"  gen_length={self.gen_length}, diffusion_steps={self.diffusion_steps}, block_length={self.block_length}, eval_batch_size={self.eval_batch_size}"
            )
            print(f"{'='*60}")

        # Split eval items across ranks (interleaved) so each rank does ~1/world_size of the work
        my_items = self.eval_items[rank::world_size]

        correct = 0
        generation_rows = []
        with torch.no_grad():
            for batch_start in range(0, len(my_items), self.eval_batch_size):
                batch = my_items[batch_start : batch_start + self.eval_batch_size]
                prompt_ids_list = [item["prompt_ids"] for item in batch]
                answer_keys = [item["answer_key"] for item in batch]
                try:
                    prompt_ids, attention_mask = self._left_pad_batch(prompt_ids_list, device)
                    out = _dllm_generate(
                        model,
                        prompt_ids,
                        self.tokenizer,
                        attention_mask=attention_mask,
                        gen_length=self.gen_length,
                        steps=self.diffusion_steps,
                        block_length=self.block_length,
                        temperature=0.0,
                        cfg_scale=0.0,
                        remasking="low_confidence",
                        mask_id=self.mask_id,
                    )
                    prompt_len = prompt_ids.shape[1]
                    for i, answer_key in enumerate(answer_keys):
                        prompt_text = self.tokenizer.decode(prompt_ids[i], skip_special_tokens=True)
                        generated = self.tokenizer.decode(out[i, prompt_len:], skip_special_tokens=False)
                        parsed = _parse_mcq_answer(generated)
                        is_correct = parsed == answer_key
                        if is_correct:
                            correct += 1
                        generation_rows.append(
                            {
                                "prompt": prompt_text,
                                "generation": generated,
                                "parsed_answer": parsed or "N/A",
                                "ground_truth": answer_key,
                                "correct": is_correct,
                            }
                        )
                except Exception as e:
                    print(f"[SciKnowEvalCallback] generation error: {e}")

        # Aggregate correct counts across all ranks
        correct_tensor = torch.tensor(correct, dtype=torch.long, device=device)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
        total_correct = correct_tensor.item()
        accuracy = total_correct / len(self.eval_items)

        # Gather all generation rows to rank 0 for saving/logging
        if dist.is_available() and dist.is_initialized():
            all_rows_per_rank = [None] * world_size
            dist.all_gather_object(all_rows_per_rank, generation_rows)
            all_generation_rows = [row for rows in all_rows_per_rank for row in rows] if rank == 0 else None
        else:
            all_generation_rows = generation_rows

        if rank == 0:
            print(f"{'='*60}")
            print(
                f"[Step {state.global_step}] ACCURACY eval done: "
                f"{accuracy:.3f} ({total_correct}/{len(self.eval_items)})"
            )
            print(f"{'='*60}\n")

            n_print = min(3, len(all_generation_rows))
            for idx, row in enumerate(all_generation_rows[:n_print]):
                print(f"  [Gen {idx+1}] GT={row['ground_truth']} | Correct={row['correct']}")
                print(f"  {row['generation']}")
                print()

            if self.output_dir is not None:
                os.makedirs(self.output_dir, exist_ok=True)
                json_path = os.path.join(self.output_dir, f"eval_generations_step_{state.global_step}.json")
                with open(json_path, "w") as f:
                    json.dump(
                        {"step": state.global_step, "accuracy": accuracy, "generations": all_generation_rows},
                        f,
                        indent=2,
                    )
                print(f"[SciKnowEvalCallback] Saved {len(all_generation_rows)} examples to {json_path}")

            try:
                import wandb

                if wandb.run is not None:
                    table = wandb.Table(
                        columns=["prompt", "generation", "parsed_answer", "ground_truth", "correct"]
                    )
                    for row in all_generation_rows:
                        table.add_data(
                            row["prompt"],
                            row["generation"],
                            row["parsed_answer"],
                            row["ground_truth"],
                            row["correct"],
                        )
                    wandb.log(
                        {
                            "eval/sciknoweval_accuracy": accuracy,
                            "eval/generations": table,
                        }
                    )
                else:
                    print("[SciKnowEvalCallback] wandb.run is None, skipping wandb log")
            except Exception as e:
                print(f"[SciKnowEvalCallback] wandb log error: {e}")

        if was_training:
            model.train()


class GSM8KEvalCallback(TrainerCallback):
    """
    Runs GSM8K boxed-answer accuracy evaluation on a fixed subset every
    `eval_steps` training steps and logs accuracy to wandb.
    """

    def __init__(
        self,
        eval_items,
        tokenizer,
        gen_length=256,
        block_length=32,
        eval_steps=50,
        eval_batch_size=4,
        output_dir=None,
        diffusion_steps=None,
        tokens_per_step=None,
        mask_id=126336,
        skip_first_step_eval=False,
    ):
        assert gen_length % block_length == 0, "gen_length must be divisible by block_length"
        self.eval_items = eval_items
        self.tokenizer = tokenizer
        self.gen_length = gen_length
        self.block_length = block_length
        self.eval_steps = eval_steps
        self.eval_batch_size = eval_batch_size
        self.output_dir = output_dir
        if diffusion_steps is not None:
            self.diffusion_steps = diffusion_steps
        elif tokens_per_step is not None:
            self.diffusion_steps = max(1, gen_length // tokens_per_step)
        else:
            self.diffusion_steps = gen_length
        self.mask_id = mask_id
        self.skip_first_step_eval = skip_first_step_eval

    def _left_pad_batch(self, prompt_ids_list, device):
        max_len = max(p.shape[0] for p in prompt_ids_list)
        pad_id = self.tokenizer.pad_token_id
        padded = torch.full((len(prompt_ids_list), max_len), pad_id, dtype=torch.long, device=device)
        attention_mask = torch.zeros(len(prompt_ids_list), max_len, dtype=torch.long, device=device)
        for i, p in enumerate(prompt_ids_list):
            padded[i, max_len - p.shape[0] :] = p.to(device)
            attention_mask[i, max_len - p.shape[0] :] = 1
        return padded, attention_mask

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        """Evaluate the base model before any training steps."""
        if self.skip_first_step_eval:
            return
        self._run_eval(args, state, model)

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step % self.eval_steps != 0:
            return
        self._run_eval(args, state, model)

    def _run_eval(self, args, state, model):
        import torch.distributed as dist

        rank = _safe_rank()
        world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1

        device = next(model.parameters()).device
        was_training = model.training
        model.eval()

        if rank == 0:
            print(f"\n{'='*60}")
            print(
                f"[Step {state.global_step}] Running ACCURACY evaluation on {len(self.eval_items)} GSM8K questions..."
            )
            print(
                f"  gen_length={self.gen_length}, diffusion_steps={self.diffusion_steps}, block_length={self.block_length}, eval_batch_size={self.eval_batch_size}"
            )
            print(f"{'='*60}")

        my_items = self.eval_items[rank::world_size]
        correct = 0
        generation_rows = []
        with torch.no_grad():
            for batch_start in range(0, len(my_items), self.eval_batch_size):
                batch = my_items[batch_start : batch_start + self.eval_batch_size]
                prompt_ids_list = [item["prompt_ids"] for item in batch]
                answer_keys = [item["answer_key"] for item in batch]
                try:
                    prompt_ids, attention_mask = self._left_pad_batch(prompt_ids_list, device)
                    out = _dllm_generate(
                        model,
                        prompt_ids,
                        self.tokenizer,
                        attention_mask=attention_mask,
                        gen_length=self.gen_length,
                        steps=self.diffusion_steps,
                        block_length=self.block_length,
                        temperature=0.0,
                        cfg_scale=0.0,
                        remasking="low_confidence",
                        mask_id=self.mask_id,
                    )
                    prompt_len = prompt_ids.shape[1]
                    for i, answer_key in enumerate(answer_keys):
                        prompt_text = self.tokenizer.decode(prompt_ids[i], skip_special_tokens=True)
                        generated = self.tokenizer.decode(out[i, prompt_len:], skip_special_tokens=False)
                        parsed = _parse_gsm8k_pred_answer(generated)
                        is_correct = _math_answers_equal(parsed, answer_key)
                        if is_correct:
                            correct += 1
                        generation_rows.append(
                            {
                                "prompt": prompt_text,
                                "generation": generated,
                                "parsed_answer": parsed or "N/A",
                                "ground_truth": answer_key,
                                "correct": is_correct,
                            }
                        )
                except Exception as e:
                    print(f"[GSM8KEvalCallback] generation error: {e}")

        correct_tensor = torch.tensor(correct, dtype=torch.long, device=device)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
        total_correct = correct_tensor.item()
        accuracy = total_correct / len(self.eval_items)

        if dist.is_available() and dist.is_initialized():
            all_rows_per_rank = [None] * world_size
            dist.all_gather_object(all_rows_per_rank, generation_rows)
            all_generation_rows = [row for rows in all_rows_per_rank for row in rows] if rank == 0 else None
        else:
            all_generation_rows = generation_rows

        if rank == 0:
            print(f"{'='*60}")
            print(
                f"[Step {state.global_step}] ACCURACY eval done: "
                f"{accuracy:.3f} ({total_correct}/{len(self.eval_items)})"
            )
            print(f"{'='*60}\n")

            n_print = min(3, len(all_generation_rows))
            for idx, row in enumerate(all_generation_rows[:n_print]):
                print(f"  [Gen {idx+1}] GT={row['ground_truth']} | Correct={row['correct']}")
                print(f"  {row['generation']}")
                print()

            if self.output_dir is not None:
                os.makedirs(self.output_dir, exist_ok=True)
                json_path = os.path.join(self.output_dir, f"eval_generations_step_{state.global_step}.json")
                with open(json_path, "w") as f:
                    json.dump(
                        {"step": state.global_step, "accuracy": accuracy, "generations": all_generation_rows},
                        f,
                        indent=2,
                    )
                print(f"[GSM8KEvalCallback] Saved {len(all_generation_rows)} examples to {json_path}")

            try:
                import wandb

                if wandb.run is not None:
                    table = wandb.Table(
                        columns=["prompt", "generation", "parsed_answer", "ground_truth", "correct"]
                    )
                    for row in all_generation_rows:
                        table.add_data(
                            row["prompt"],
                            row["generation"],
                            row["parsed_answer"],
                            row["ground_truth"],
                            row["correct"],
                        )
                    wandb.log(
                        {
                            "eval/gsm8k_accuracy": accuracy,
                            "eval/generations": table,
                        }
                    )
                else:
                    print("[GSM8KEvalCallback] wandb.run is None, skipping wandb log")
            except Exception as e:
                print(f"[GSM8KEvalCallback] wandb log error: {e}")

        if was_training:
            model.train()


class MATH500EvalCallback(GSM8KEvalCallback):
    """
    Runs MATH-500 boxed-answer accuracy evaluation every `eval_steps` training steps.

    Uses is_equiv (via _math_answers_equal) for symbolic equivalence checking,
    consistent with diffu-grpo's correctness_reward_func_math. Logs to wandb
    under eval/math500_accuracy.
    """

    def _run_eval(self, args, state, model):
        import torch.distributed as dist

        rank = _safe_rank()
        world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1

        device = next(model.parameters()).device
        was_training = model.training
        model.eval()

        if rank == 0:
            print(f"\n{'='*60}")
            print(
                f"[Step {state.global_step}] Running ACCURACY evaluation on {len(self.eval_items)} MATH-500 questions..."
            )
            print(
                f"  gen_length={self.gen_length}, diffusion_steps={self.diffusion_steps}, block_length={self.block_length}, eval_batch_size={self.eval_batch_size}"
            )
            print(f"{'='*60}")

        my_items = self.eval_items[rank::world_size]
        correct = 0
        generation_rows = []
        with torch.no_grad():
            for batch_start in range(0, len(my_items), self.eval_batch_size):
                batch = my_items[batch_start : batch_start + self.eval_batch_size]
                prompt_ids_list = [item["prompt_ids"] for item in batch]
                answer_keys = [item["answer_key"] for item in batch]
                try:
                    prompt_ids, attention_mask = self._left_pad_batch(prompt_ids_list, device)
                    out = _dllm_generate(
                        model,
                        prompt_ids,
                        self.tokenizer,
                        attention_mask=attention_mask,
                        gen_length=self.gen_length,
                        steps=self.diffusion_steps,
                        block_length=self.block_length,
                        temperature=0.0,
                        cfg_scale=0.0,
                        remasking="low_confidence",
                        mask_id=self.mask_id,
                    )
                    prompt_len = prompt_ids.shape[1]
                    for i, answer_key in enumerate(answer_keys):
                        prompt_text = self.tokenizer.decode(prompt_ids[i], skip_special_tokens=True)
                        generated = self.tokenizer.decode(out[i, prompt_len:], skip_special_tokens=False)
                        try:
                            parsed = remove_boxed(last_boxed_only_string(generated))
                        except Exception:
                            parsed = None
                        if not parsed:
                            m = re.search(r"<answer>(.*?)</answer>", generated, re.DOTALL)
                            parsed = m.group(1).strip() if m else None
                        is_correct = _ph_is_equiv(parsed, answer_key) if parsed is not None else False
                        if is_correct:
                            correct += 1
                        generation_rows.append(
                            {
                                "prompt": prompt_text,
                                "generation": generated,
                                "parsed_answer": parsed or "N/A",
                                "ground_truth": answer_key,
                                "correct": is_correct,
                            }
                        )
                except Exception as e:
                    print(f"[MATH500EvalCallback] generation error: {e}")

        correct_tensor = torch.tensor(correct, dtype=torch.long, device=device)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
        total_correct = correct_tensor.item()
        accuracy = total_correct / len(self.eval_items)

        if dist.is_available() and dist.is_initialized():
            all_rows_per_rank = [None] * world_size
            dist.all_gather_object(all_rows_per_rank, generation_rows)
            all_generation_rows = [row for rows in all_rows_per_rank for row in rows] if rank == 0 else None
        else:
            all_generation_rows = generation_rows

        if rank == 0:
            print(f"{'='*60}")
            print(
                f"[Step {state.global_step}] MATH-500 ACCURACY eval done: "
                f"{accuracy:.3f} ({total_correct}/{len(self.eval_items)})"
            )
            print(f"{'='*60}\n")

            n_print = min(3, len(all_generation_rows))
            for idx, row in enumerate(all_generation_rows[:n_print]):
                print(f"  [Gen {idx+1}] GT={row['ground_truth']} | Correct={row['correct']}")
                print(f"  {row['generation']}")
                print()

            if self.output_dir is not None:
                os.makedirs(self.output_dir, exist_ok=True)
                json_path = os.path.join(self.output_dir, f"eval_math500_step_{state.global_step}.json")
                with open(json_path, "w") as f:
                    json.dump(
                        {"step": state.global_step, "accuracy": accuracy, "generations": all_generation_rows},
                        f,
                        indent=2,
                    )
                print(f"[MATH500EvalCallback] Saved {len(all_generation_rows)} examples to {json_path}")

            try:
                import wandb

                if wandb.run is not None:
                    table = wandb.Table(
                        columns=["prompt", "generation", "parsed_answer", "ground_truth", "correct"]
                    )
                    for row in all_generation_rows:
                        table.add_data(
                            row["prompt"],
                            row["generation"],
                            row["parsed_answer"],
                            row["ground_truth"],
                            row["correct"],
                        )
                    wandb.log(
                        {
                            "eval/math500_accuracy": accuracy,
                            "eval/math500_generations": table,
                        }
                    )
                else:
                    print("[MATH500EvalCallback] wandb.run is None, skipping wandb log")
            except Exception as e:
                print(f"[MATH500EvalCallback] wandb log error: {e}")

        if was_training:
            model.train()


class CodeEvalCallback(TrainerCallback):
    """
    Runs MBPP pass@1 accuracy evaluation every `eval_steps` training steps.

    Uses Parser.extract_answer_code + test_solution from eval/parsers.py,
    matching the evaluation process in eval/mbpp.py.
    """

    def __init__(
        self,
        eval_items,
        tokenizer,
        gen_length=512,
        block_length=32,
        eval_steps=50,
        eval_batch_size=4,
        output_dir=None,
        diffusion_steps=None,
        tokens_per_step=None,
        mask_id=126336,
        skip_first_step_eval=False,
    ):
        assert gen_length % block_length == 0, "gen_length must be divisible by block_length"
        self.eval_items = eval_items
        self.tokenizer = tokenizer
        self.gen_length = gen_length
        self.block_length = block_length
        self.eval_steps = eval_steps
        self.eval_batch_size = eval_batch_size
        self.output_dir = output_dir
        if diffusion_steps is not None:
            self.diffusion_steps = diffusion_steps
        elif tokens_per_step is not None:
            self.diffusion_steps = max(1, gen_length // tokens_per_step)
        else:
            self.diffusion_steps = gen_length
        self.mask_id = mask_id
        self.skip_first_step_eval = skip_first_step_eval

    def _left_pad_batch(self, prompt_ids_list, device):
        max_len = max(p.shape[0] for p in prompt_ids_list)
        pad_id = self.tokenizer.pad_token_id
        padded = torch.full((len(prompt_ids_list), max_len), pad_id, dtype=torch.long, device=device)
        attention_mask = torch.zeros(len(prompt_ids_list), max_len, dtype=torch.long, device=device)
        for i, p in enumerate(prompt_ids_list):
            padded[i, max_len - p.shape[0] :] = p.to(device)
            attention_mask[i, max_len - p.shape[0] :] = 1
        return padded, attention_mask

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if self.skip_first_step_eval:
            return
        self._run_eval(args, state, model)

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step % self.eval_steps != 0:
            return
        self._run_eval(args, state, model)

    def _run_eval(self, args, state, model):
        import torch.distributed as dist

        Parser, test_solution = _get_code_eval_helpers()

        rank = _safe_rank()
        world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1

        device = next(model.parameters()).device
        was_training = model.training
        model.eval()

        if rank == 0:
            print(f"\n{'='*60}")
            print(f"[Step {state.global_step}] Running MBPP eval on {len(self.eval_items)} questions...")
            print(
                f"  gen_length={self.gen_length}, diffusion_steps={self.diffusion_steps}, block_length={self.block_length}"
            )
            print(f"{'='*60}")

        my_items = self.eval_items[rank::world_size]
        correct = 0
        generation_rows = []
        with torch.no_grad():
            for batch_start in range(0, len(my_items), self.eval_batch_size):
                batch = my_items[batch_start : batch_start + self.eval_batch_size]
                prompt_ids_list = [item["prompt_ids"] for item in batch]
                try:
                    prompt_ids, attention_mask = self._left_pad_batch(prompt_ids_list, device)
                    out = _dllm_generate(
                        model,
                        prompt_ids,
                        self.tokenizer,
                        attention_mask=attention_mask,
                        gen_length=self.gen_length,
                        steps=self.diffusion_steps,
                        block_length=self.block_length,
                        temperature=0.0,
                        cfg_scale=0.0,
                        remasking="low_confidence",
                        mask_id=self.mask_id,
                    )
                    prompt_len = prompt_ids.shape[1]
                    for i, item in enumerate(batch):
                        prompt_text = self.tokenizer.decode(prompt_ids[i], skip_special_tokens=True)
                        generated = self.tokenizer.decode(out[i, prompt_len:], skip_special_tokens=False)
                        # Follow mbpp.py: extract code then run code + unit tests
                        program = Parser.extract_answer_code(generated)
                        is_correct = False
                        if program is not None:
                            test_code = program + "\n\n" + item["test_code"]
                            is_correct = bool(test_solution(test_code, self.output_dir))
                        if is_correct:
                            correct += 1
                        generation_rows.append(
                            {
                                "prompt": prompt_text,
                                "generation": generated,
                                "extracted_code": program or "N/A",
                                "correct": is_correct,
                            }
                        )
                except Exception as e:
                    print(f"[CodeEvalCallback] generation error: {e}")

        correct_tensor = torch.tensor(correct, dtype=torch.long, device=device)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
        total_correct = correct_tensor.item()
        accuracy = total_correct / len(self.eval_items)

        if dist.is_available() and dist.is_initialized():
            all_rows_per_rank = [None] * world_size
            dist.all_gather_object(all_rows_per_rank, generation_rows)
            all_generation_rows = [row for rows in all_rows_per_rank for row in rows] if rank == 0 else None
        else:
            all_generation_rows = generation_rows

        if rank == 0:
            print(f"{'='*60}")
            print(
                f"[Step {state.global_step}] MBPP eval done: "
                f"{accuracy:.3f} ({total_correct}/{len(self.eval_items)})"
            )
            print(f"{'='*60}\n")

            n_print = min(3, len(all_generation_rows))
            for idx, row in enumerate(all_generation_rows[:n_print]):
                print(f"  [Gen {idx+1}] Correct={row['correct']}")
                print(f"  {row['generation']}")
                print()

            if self.output_dir is not None:
                os.makedirs(self.output_dir, exist_ok=True)
                json_path = os.path.join(self.output_dir, f"eval_generations_step_{state.global_step}.json")
                with open(json_path, "w") as f:
                    json.dump(
                        {"step": state.global_step, "accuracy": accuracy, "generations": all_generation_rows},
                        f,
                        indent=2,
                    )
                print(f"[CodeEvalCallback] Saved {len(all_generation_rows)} examples to {json_path}")

            try:
                import wandb

                if wandb.run is not None:
                    table = wandb.Table(columns=["prompt", "generation", "extracted_code", "correct"])
                    for row in all_generation_rows:
                        table.add_data(
                            row["prompt"],
                            row["generation"],
                            row["extracted_code"],
                            row["correct"],
                        )
                    wandb.log(
                        {
                            "eval/mbpp_accuracy": accuracy,
                            "eval/generations": table,
                        }
                    )
                else:
                    print("[CodeEvalCallback] wandb.run is None, skipping wandb log")
            except Exception as e:
                print(f"[CodeEvalCallback] wandb log error: {e}")

        if was_training:
            model.train()


class HumanEvalEvalCallback(TrainerCallback):
    """
    Runs HumanEval pass@1 accuracy evaluation every `eval_steps` training steps.

    Uses Parser.extract_answer_code + test_solution from eval/parsers.py,
    matching the evaluation process in eval/human_eval.py.
    """

    def __init__(
        self,
        eval_items,
        tokenizer,
        gen_length=512,
        block_length=32,
        eval_steps=50,
        eval_batch_size=4,
        output_dir=None,
        diffusion_steps=None,
        tokens_per_step=None,
        mask_id=126336,
        skip_first_step_eval=False,
    ):
        assert gen_length % block_length == 0, "gen_length must be divisible by block_length"
        self.eval_items = eval_items
        self.tokenizer = tokenizer
        self.gen_length = gen_length
        self.block_length = block_length
        self.eval_steps = eval_steps
        self.eval_batch_size = eval_batch_size
        self.output_dir = output_dir
        if diffusion_steps is not None:
            self.diffusion_steps = diffusion_steps
        elif tokens_per_step is not None:
            self.diffusion_steps = max(1, gen_length // tokens_per_step)
        else:
            self.diffusion_steps = gen_length
        self.mask_id = mask_id
        self.skip_first_step_eval = skip_first_step_eval

    def _left_pad_batch(self, prompt_ids_list, device):
        max_len = max(p.shape[0] for p in prompt_ids_list)
        pad_id = self.tokenizer.pad_token_id
        padded = torch.full((len(prompt_ids_list), max_len), pad_id, dtype=torch.long, device=device)
        attention_mask = torch.zeros(len(prompt_ids_list), max_len, dtype=torch.long, device=device)
        for i, p in enumerate(prompt_ids_list):
            padded[i, max_len - p.shape[0] :] = p.to(device)
            attention_mask[i, max_len - p.shape[0] :] = 1
        return padded, attention_mask

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if self.skip_first_step_eval:
            return
        self._run_eval(args, state, model)

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step % self.eval_steps != 0:
            return
        self._run_eval(args, state, model)

    def _run_eval(self, args, state, model):
        import re as _re
        import torch.distributed as dist

        Parser, test_solution, _ = _get_code_eval_helpers()

        rank = _safe_rank()
        world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1

        device = next(model.parameters()).device
        was_training = model.training
        model.eval()

        if rank == 0:
            print(f"\n{'='*60}")
            print(f"[Step {state.global_step}] Running HumanEval eval on {len(self.eval_items)} questions...")
            print(
                f"  gen_length={self.gen_length}, diffusion_steps={self.diffusion_steps}, block_length={self.block_length}"
            )
            print(f"{'='*60}")

        if self.output_dir is not None:
            os.makedirs(self.output_dir, exist_ok=True)

        my_items = self.eval_items[rank::world_size]
        correct = 0
        generation_rows = []
        with torch.no_grad():
            for batch_start in range(0, len(my_items), self.eval_batch_size):
                batch = my_items[batch_start : batch_start + self.eval_batch_size]
                prompt_ids_list = [item["prompt_ids"] for item in batch]
                try:
                    prompt_ids, attention_mask = self._left_pad_batch(prompt_ids_list, device)
                    out = _dllm_generate(
                        model,
                        prompt_ids,
                        self.tokenizer,
                        attention_mask=attention_mask,
                        gen_length=self.gen_length,
                        steps=self.diffusion_steps,
                        block_length=self.block_length,
                        temperature=0.0,
                        cfg_scale=0.0,
                        remasking="low_confidence",
                        mask_id=self.mask_id,
                    )
                    prompt_len = prompt_ids.shape[1]
                    for i, item in enumerate(batch):
                        prompt_text = self.tokenizer.decode(prompt_ids[i], skip_special_tokens=True)
                        generated = self.tokenizer.decode(out[i, prompt_len:], skip_special_tokens=False)
                        # Follow human_eval.py: extract code, build test_code with check(defined_func)
                        program = Parser.extract_answer_code(generated)
                        is_correct = False
                        if program is not None:
                            solution_match = _re.search(r"def (\w+)\(", program)
                            if solution_match is not None:
                                defined_func = solution_match.group(1)
                                test_str = item["test"]
                                unit_test = test_str[test_str.index("def") :]
                                test_code = program + "\n\n" + unit_test + "\n\n" + f"check({defined_func})"
                                is_correct = bool(test_solution(test_code, self.output_dir))
                        if is_correct:
                            correct += 1
                        generation_rows.append(
                            {
                                "prompt": prompt_text,
                                "generation": generated,
                                "extracted_code": program or "N/A",
                                "correct": is_correct,
                            }
                        )
                except Exception as e:
                    print(f"[HumanEvalEvalCallback] generation error: {e}")

        correct_tensor = torch.tensor(correct, dtype=torch.long, device=device)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
        total_correct = correct_tensor.item()
        accuracy = total_correct / len(self.eval_items)

        if dist.is_available() and dist.is_initialized():
            all_rows_per_rank = [None] * world_size
            dist.all_gather_object(all_rows_per_rank, generation_rows)
            all_generation_rows = [row for rows in all_rows_per_rank for row in rows] if rank == 0 else None
        else:
            all_generation_rows = generation_rows

        if rank == 0:
            print(f"{'='*60}")
            print(
                f"[Step {state.global_step}] HumanEval eval done: "
                f"{accuracy:.3f} ({total_correct}/{len(self.eval_items)})"
            )
            print(f"{'='*60}\n")

            if self.output_dir is not None:
                os.makedirs(self.output_dir, exist_ok=True)
                json_path = os.path.join(self.output_dir, f"humaneval_step_{state.global_step}.json")
                with open(json_path, "w") as f:
                    json.dump(
                        {"step": state.global_step, "accuracy": accuracy, "generations": all_generation_rows},
                        f,
                        indent=2,
                    )
                print(f"[HumanEvalEvalCallback] Saved {len(all_generation_rows)} examples to {json_path}")

            try:
                import wandb

                if wandb.run is not None:
                    table = wandb.Table(columns=["prompt", "generation", "extracted_code", "correct"])
                    for row in all_generation_rows:
                        table.add_data(
                            row["prompt"],
                            row["generation"],
                            row["extracted_code"],
                            row["correct"],
                        )
                    wandb.log(
                        {
                            "eval/humaneval_accuracy": accuracy,
                            "eval/humaneval_generations": table,
                        }
                    )
                else:
                    print("[HumanEvalEvalCallback] wandb.run is None, skipping wandb log")
            except Exception as e:
                print(f"[HumanEvalEvalCallback] wandb log error: {e}")

        if was_training:
            model.train()


# ---------------------------------------------------------------------------
# Sudoku support
# ---------------------------------------------------------------------------

SUDOKU_SYSTEM_PROMPT_SFT = """Please solve the following 4x4 Sudoku puzzle. The puzzle is provided as a 16-character string reading left-to-right, top-to-bottom, where '0' represents empty cells.

Rules:
- Fill empty cells with digits 1-4
- Each row must contain digits 1-4 exactly once
- Each column must contain digits 1-4 exactly once
- Each 2x2 box must contain digits 1-4 exactly once

Important: Your solution must be a COMPLETE 16-character string with only the digits 1-4, representing your final solved grid.

Respond in this exact format:
<reasoning>
Your step-by-step solving process
</reasoning>
<answer>
[16-character solution string with no spaces or separators]
</answer>"""


def _extract_sudoku_answer(text):
    """Extract 16-digit solution using the same multi-pattern approach as
    parse_sudoku_answers() in eval/parser_json.py."""
    solution_str = ""
    patterns = [
        r"<answer>.*?```\s*([\d\s]+)```",
        r"<answer>(.*?)(?:<\|eot_id\|>|<\|endoftext\|>|</answer>)",
        r"</answer>\s*(.*?)(?:<\|eot_id\|>|<\|endoftext\|>|$)",
        r".*?(\d{16})\s*</answer>",
        r"\b(\d{16})\b",
    ]
    for pattern in patterns:
        if solution_str:
            break
        match = re.search(pattern, text, re.DOTALL)
        if match and match.group(1).strip():
            solution_str = match.group(1).strip()
    solution_str = re.sub(r"\s", "", solution_str)
    return solution_str if solution_str else None


def _score_sudoku(prediction, ground_truth, puzzle):
    """Returns (correct_cells, empty_cells) matching the aggregation used by
    parse_sudoku_answers() in eval/parser_json.py.

    The overall cell accuracy is total_correct_cells / total_empty_cells
    (weighted aggregate), not the mean of per-puzzle fractions.
    """
    empty_indices = [i for i in range(16) if puzzle[i] == "0"]
    empty_cells = len(empty_indices)
    if not empty_indices:
        return 0, 0
    if prediction is None or len(prediction) == 0:
        return 0, empty_cells
    if len(prediction) < 16:
        prediction = prediction + "0" * (16 - len(prediction))
    elif len(prediction) > 16:
        prediction = prediction[:16]
    correct = sum(1 for i in empty_indices if prediction[i] == ground_truth[i])
    return correct, empty_cells


def preprocess_sudoku_for_self_distillation(
    data,
    tokenizer,
    max_length,
    min_reasoning_tokens=128,
    max_reasoning_tokens=256,
    test_split=0.1,
    cache_dir=None,
):
    """Preprocess sudoku data (Puzzle/Solution columns) for self-distillation."""
    items = []
    for i in range(len(data)):
        puzzle = data[i]["Puzzle"]
        solution = data[i]["Solution"]
        question_text = SUDOKU_SYSTEM_PROMPT_SFT + f"\n\nSolve the following Sudoku puzzle: {puzzle}\n"
        answer_text = f"</reasoning>\n<answer>{solution}</answer>"
        items.append({"question_text": question_text, "answer_text": answer_text})

    random.shuffle(items)
    split = int(len(items) * test_split)
    train_items, test_items = items[split:], items[:split]

    train_data = _preprocess_masked_reasoning(
        train_items,
        tokenizer,
        max_length,
        min_reasoning_tokens,
        max_reasoning_tokens,
        desc="Preprocessing Sudoku train",
        cache_dir=cache_dir,
    )
    test_data = _preprocess_masked_reasoning(
        test_items,
        tokenizer,
        max_length,
        min_reasoning_tokens,
        max_reasoning_tokens,
        desc="Preprocessing Sudoku eval",
        cache_dir=cache_dir,
    )
    return train_data, test_data


def build_sudoku_eval_prompts(tokenizer, n=256, max_prompt_length=512, seed=42):
    """Sample n examples from 4x4_test_sudoku.csv for callback eval."""
    import pandas as pd

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.normpath(os.path.join(cur_dir, "..", "dataset", "4x4_test_sudoku.csv"))
    df = pd.read_csv(csv_path, dtype={"Puzzle": str, "Solution": str})
    indices = random.Random(seed).sample(range(len(df)), min(n, len(df)))
    eval_items = []
    for i in indices:
        row = df.iloc[i]
        puzzle = row["Puzzle"]
        solution = row["Solution"]
        question_text = SUDOKU_SYSTEM_PROMPT_SFT + f"\n\nSolve the following Sudoku puzzle: {puzzle}\n"
        prompt_msgs = [{"role": "user", "content": question_text}]
        prompt_text = (
            tokenizer.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
            + "<reasoning>"
        )
        prompt_ids = tokenizer(
            prompt_text, return_tensors="pt", truncation=True, max_length=max_prompt_length
        ).input_ids.squeeze(0)
        eval_items.append({"prompt_ids": prompt_ids, "puzzle": puzzle, "solution": solution})
    return eval_items


class SudokuEvalCallback(TrainerCallback):
    """
    Runs Sudoku cell-accuracy evaluation on a fixed subset every `eval_steps` steps.

    Uses 4x4_test_sudoku.csv as the eval set. Logs mean cell accuracy (fraction of
    empty cells filled correctly) and exact-match accuracy to wandb.
    """

    def __init__(
        self,
        eval_items,
        tokenizer,
        gen_length=256,
        block_length=32,
        eval_steps=50,
        eval_batch_size=4,
        output_dir=None,
        diffusion_steps=None,
        tokens_per_step=None,
        mask_id=126336,
        skip_first_step_eval=False,
    ):
        assert gen_length % block_length == 0, "gen_length must be divisible by block_length"
        self.eval_items = eval_items
        self.tokenizer = tokenizer
        self.gen_length = gen_length
        self.block_length = block_length
        self.eval_steps = eval_steps
        self.eval_batch_size = eval_batch_size
        self.output_dir = output_dir
        if diffusion_steps is not None:
            self.diffusion_steps = diffusion_steps
        elif tokens_per_step is not None:
            self.diffusion_steps = max(1, gen_length // tokens_per_step)
        else:
            self.diffusion_steps = gen_length
        self.mask_id = mask_id
        self.skip_first_step_eval = skip_first_step_eval

    def _left_pad_batch(self, prompt_ids_list, device):
        max_len = max(p.shape[0] for p in prompt_ids_list)
        pad_id = self.tokenizer.pad_token_id
        padded = torch.full((len(prompt_ids_list), max_len), pad_id, dtype=torch.long, device=device)
        attention_mask = torch.zeros(len(prompt_ids_list), max_len, dtype=torch.long, device=device)
        for i, p in enumerate(prompt_ids_list):
            padded[i, max_len - p.shape[0] :] = p.to(device)
            attention_mask[i, max_len - p.shape[0] :] = 1
        return padded, attention_mask

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if self.skip_first_step_eval:
            return
        self._run_eval(args, state, model)

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step % self.eval_steps != 0:
            return
        self._run_eval(args, state, model)

    def _run_eval(self, args, state, model):
        import torch.distributed as dist

        rank = _safe_rank()
        world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1

        device = next(model.parameters()).device
        was_training = model.training
        model.eval()

        if rank == 0:
            print(f"\n{'='*60}")
            print(f"[Step {state.global_step}] Running SUDOKU eval on {len(self.eval_items)} puzzles...")
            print(
                f"  gen_length={self.gen_length}, diffusion_steps={self.diffusion_steps}, block_length={self.block_length}"
            )
            print(f"{'='*60}")

        my_items = self.eval_items[rank::world_size]
        total_correct_cells = 0
        total_empty_cells = 0
        exact_correct = 0
        generation_rows = []
        with torch.no_grad():
            for batch_start in range(0, len(my_items), self.eval_batch_size):
                batch = my_items[batch_start : batch_start + self.eval_batch_size]
                prompt_ids_list = [item["prompt_ids"] for item in batch]
                try:
                    prompt_ids, attention_mask = self._left_pad_batch(prompt_ids_list, device)
                    out = _dllm_generate(
                        model,
                        prompt_ids,
                        self.tokenizer,
                        attention_mask=attention_mask,
                        gen_length=self.gen_length,
                        steps=self.diffusion_steps,
                        block_length=self.block_length,
                        temperature=0.0,
                        cfg_scale=0.0,
                        remasking="low_confidence",
                        mask_id=self.mask_id,
                    )
                    prompt_len = prompt_ids.shape[1]
                    for i, item in enumerate(batch):
                        prompt_text = self.tokenizer.decode(prompt_ids[i], skip_special_tokens=True)
                        generated = self.tokenizer.decode(out[i, prompt_len:], skip_special_tokens=False)
                        parsed = _extract_sudoku_answer(generated)
                        correct_cells, empty_cells = _score_sudoku(parsed, item["solution"], item["puzzle"])
                        total_correct_cells += correct_cells
                        total_empty_cells += empty_cells
                        cell_score = correct_cells / empty_cells if empty_cells > 0 else 0.0
                        is_exact = cell_score == 1.0
                        if is_exact:
                            exact_correct += 1
                        generation_rows.append(
                            {
                                "prompt": prompt_text,
                                "generation": generated,
                                "parsed_answer": parsed or "N/A",
                                "ground_truth": item["solution"],
                                "puzzle": item["puzzle"],
                                "cell_score": cell_score,
                                "exact": is_exact,
                            }
                        )
                except Exception as e:
                    print(f"[SudokuEvalCallback] generation error: {e}")

        # Use the same aggregation as parse_sudoku_answers() in eval/parser_json.py:
        # total_correct_cells / total_empty_cells (weighted across puzzles).
        correct_tensor = torch.tensor(total_correct_cells, dtype=torch.float, device=device)
        empty_tensor = torch.tensor(total_empty_cells, dtype=torch.float, device=device)
        exact_tensor = torch.tensor(exact_correct, dtype=torch.long, device=device)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(empty_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(exact_tensor, op=dist.ReduceOp.SUM)
        mean_cell_score = correct_tensor.item() / empty_tensor.item() if empty_tensor.item() > 0 else 0.0
        exact_accuracy = exact_tensor.item() / len(self.eval_items)

        if dist.is_available() and dist.is_initialized():
            all_rows_per_rank = [None] * world_size
            dist.all_gather_object(all_rows_per_rank, generation_rows)
            all_generation_rows = [row for rows in all_rows_per_rank for row in rows] if rank == 0 else None
        else:
            all_generation_rows = generation_rows

        if rank == 0:
            print(f"{'='*60}")
            print(
                f"[Step {state.global_step}] SUDOKU eval done: "
                f"cell_score={mean_cell_score:.3f}, exact={exact_accuracy:.3f} ({exact_tensor.item()}/{len(self.eval_items)})"
            )
            print(f"{'='*60}\n")

            n_print = min(3, len(all_generation_rows))
            for idx, row in enumerate(all_generation_rows[:n_print]):
                print(f"  [Gen {idx+1}] CellScore={row['cell_score']:.2f} | Exact={row['exact']}")
                print(f"  {row['generation']}")
                print()

            if self.output_dir is not None:
                os.makedirs(self.output_dir, exist_ok=True)
                json_path = os.path.join(self.output_dir, f"eval_sudoku_step_{state.global_step}.json")
                with open(json_path, "w") as f:
                    json.dump(
                        {
                            "step": state.global_step,
                            "mean_cell_score": mean_cell_score,
                            "total_correct_cells": int(correct_tensor.item()),
                            "total_empty_cells": int(empty_tensor.item()),
                            "exact_accuracy": exact_accuracy,
                            "generations": all_generation_rows,
                        },
                        f,
                        indent=2,
                    )
                print(f"[SudokuEvalCallback] Saved {len(all_generation_rows)} examples to {json_path}")

            try:
                import wandb

                if wandb.run is not None:
                    table = wandb.Table(
                        columns=[
                            "prompt",
                            "generation",
                            "parsed_answer",
                            "ground_truth",
                            "puzzle",
                            "cell_score",
                            "exact",
                        ]
                    )
                    for row in all_generation_rows:
                        table.add_data(
                            row["prompt"],
                            row["generation"],
                            row["parsed_answer"],
                            row["ground_truth"],
                            row["puzzle"],
                            row["cell_score"],
                            row["exact"],
                        )
                    wandb.log(
                        {
                            "eval/sudoku_cell_score": mean_cell_score,
                            "eval/sudoku_exact_accuracy": exact_accuracy,
                            "eval/sudoku_generations": table,
                        }
                    )
                else:
                    print("[SudokuEvalCallback] wandb.run is None, skipping wandb log")
            except Exception as e:
                print(f"[SudokuEvalCallback] wandb log error: {e}")

        if was_training:
            model.train()


# ---------------------------------------------------------------------------
# Countdown support
# ---------------------------------------------------------------------------

COUNTDOWN_SYSTEM_PROMPT_SFT = """Using only the provided numbers, create an arithmetic expression that evaluates to exactly the provided target number. You may use the operations +, -, *, and / as needed, but each number must be used exactly once.

Respond in the following format:
<reasoning>
Your reasoning here
</reasoning>
<answer>
your arithmetic expression (e.g. 2 + 3 * 4)
</answer>"""


def _solve_countdown(numbers, target):
    """Brute-force find an expression using all 3 numbers (exactly once) equalling target."""
    from itertools import permutations, product as iproduct

    ops = ["+", "-", "*", "/"]
    for perm in permutations(numbers):
        a, b, c = perm
        for op1, op2 in iproduct(ops, ops):
            for expr in [f"({a} {op1} {b}) {op2} {c}", f"{a} {op1} ({b} {op2} {c})"]:
                try:
                    result = eval(expr, {"__builtins__": None}, {})  # noqa: S307
                    if result is not None and abs(result - target) < 1e-9:
                        return expr
                except Exception:
                    continue
    return None


def _extract_countdown_answer(text):
    """Extract arithmetic expression from <answer> tags."""
    matches = re.findall(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if matches:
        return matches[-1].strip()
    return None


def _score_countdown(prediction, numbers, target):
    """Returns 1.0 if prediction is a valid expression using all numbers equalling target."""
    if prediction is None or prediction.strip() == "":
        return 0.0
    try:
        nums_in_pred = [int(n) for n in re.findall(r"\d+", prediction)]
        if sorted(nums_in_pred) != sorted(int(x) for x in numbers):
            return 0.0
        if not re.match(r"^[\d+\-*/().\s]+$", prediction):
            return 0.0
        result = eval(prediction, {"__builtins__": None}, {})  # noqa: S307
        if result is not None and abs(result - target) < 1e-5:
            return 1.0
    except Exception:
        pass
    return 0.0


def preprocess_countdown_for_self_distillation(
    data,
    tokenizer,
    max_length,
    min_reasoning_tokens=128,
    max_reasoning_tokens=256,
    test_split=0.1,
    cache_dir=None,
):
    """Preprocess Countdown-Tasks dataset for self-distillation.

    Brute-force solves each puzzle to obtain a reference expression; examples
    with no solution are skipped.
    """
    items = []
    skipped = 0
    for i in tqdm(range(len(data)), desc="Solving Countdown puzzles"):
        numbers = list(data[i]["nums"])
        target = int(data[i]["target"])
        solution = _solve_countdown(numbers, target)
        if solution is None:
            skipped += 1
            continue
        question_text = COUNTDOWN_SYSTEM_PROMPT_SFT + f"\n\nNumbers: {numbers}\nTarget: {target}"
        answer_text = f"</reasoning>\n<answer>{solution}</answer>"
        items.append({"question_text": question_text, "answer_text": answer_text})

    if skipped > 0:
        print(f"[Countdown] Skipped {skipped} unsolvable examples out of {len(data)}")

    random.shuffle(items)
    split = int(len(items) * test_split)
    train_items, test_items = items[split:], items[:split]

    train_data = _preprocess_masked_reasoning(
        train_items,
        tokenizer,
        max_length,
        min_reasoning_tokens,
        max_reasoning_tokens,
        desc="Preprocessing Countdown train",
        cache_dir=cache_dir,
    )
    test_data = _preprocess_masked_reasoning(
        test_items,
        tokenizer,
        max_length,
        min_reasoning_tokens,
        max_reasoning_tokens,
        desc="Preprocessing Countdown eval",
        cache_dir=cache_dir,
    )
    return train_data, test_data


def build_countdown_eval_prompts(tokenizer, n=256, max_prompt_length=512, seed=42):
    """Sample n examples from countdown_cd3_test.jsonl for callback eval."""
    import json as _json

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    jsonl_path = os.path.normpath(os.path.join(cur_dir, "..", "dataset", "countdown_cd3_test.jsonl"))
    dataset = []
    with open(jsonl_path, "r") as f:
        for line in f:
            dataset.append(_json.loads(line))
    indices = random.Random(seed).sample(range(len(dataset)), min(n, len(dataset)))
    eval_items = []
    for i in indices:
        entry = dataset[i]
        target = int(entry["output"])
        numbers = [int(x) for x in entry["input"].split(",")]
        question_text = COUNTDOWN_SYSTEM_PROMPT_SFT + f"\n\nNumbers: {numbers}\nTarget: {target}"
        prompt_msgs = [{"role": "user", "content": question_text}]
        prompt_text = tokenizer.apply_chat_template(prompt_msgs, tokenize=False) + "\n"
        prompt_ids = tokenizer(
            prompt_text, return_tensors="pt", truncation=True, max_length=max_prompt_length
        ).input_ids.squeeze(0)
        eval_items.append({"prompt_ids": prompt_ids, "numbers": numbers, "target": target})
    return eval_items


class CountdownEvalCallback(TrainerCallback):
    """
    Runs Countdown exact-match accuracy evaluation on a fixed subset every `eval_steps` steps.

    Uses countdown_cd3_test.jsonl as the eval set. Logs accuracy (fraction of
    expressions that evaluate to the correct target) to wandb.
    """

    def __init__(
        self,
        eval_items,
        tokenizer,
        gen_length=256,
        block_length=32,
        eval_steps=50,
        eval_batch_size=4,
        output_dir=None,
        diffusion_steps=None,
        tokens_per_step=None,
        mask_id=126336,
        skip_first_step_eval=False,
    ):
        assert gen_length % block_length == 0, "gen_length must be divisible by block_length"
        self.eval_items = eval_items
        self.tokenizer = tokenizer
        self.gen_length = gen_length
        self.block_length = block_length
        self.eval_steps = eval_steps
        self.eval_batch_size = eval_batch_size
        self.output_dir = output_dir
        if diffusion_steps is not None:
            self.diffusion_steps = diffusion_steps
        elif tokens_per_step is not None:
            self.diffusion_steps = max(1, gen_length // tokens_per_step)
        else:
            self.diffusion_steps = gen_length
        self.mask_id = mask_id
        self.skip_first_step_eval = skip_first_step_eval

    def _left_pad_batch(self, prompt_ids_list, device):
        max_len = max(p.shape[0] for p in prompt_ids_list)
        pad_id = self.tokenizer.pad_token_id
        padded = torch.full((len(prompt_ids_list), max_len), pad_id, dtype=torch.long, device=device)
        attention_mask = torch.zeros(len(prompt_ids_list), max_len, dtype=torch.long, device=device)
        for i, p in enumerate(prompt_ids_list):
            padded[i, max_len - p.shape[0] :] = p.to(device)
            attention_mask[i, max_len - p.shape[0] :] = 1
        return padded, attention_mask

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if self.skip_first_step_eval:
            return
        self._run_eval(args, state, model)

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step % self.eval_steps != 0:
            return
        self._run_eval(args, state, model)

    def _run_eval(self, args, state, model):
        import torch.distributed as dist

        rank = _safe_rank()
        world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1

        device = next(model.parameters()).device
        was_training = model.training
        model.eval()

        if rank == 0:
            print(f"\n{'='*60}")
            print(f"[Step {state.global_step}] Running COUNTDOWN eval on {len(self.eval_items)} puzzles...")
            print(
                f"  gen_length={self.gen_length}, diffusion_steps={self.diffusion_steps}, block_length={self.block_length}"
            )
            print(f"{'='*60}")

        my_items = self.eval_items[rank::world_size]
        correct = 0
        generation_rows = []
        with torch.no_grad():
            for batch_start in range(0, len(my_items), self.eval_batch_size):
                batch = my_items[batch_start : batch_start + self.eval_batch_size]
                prompt_ids_list = [item["prompt_ids"] for item in batch]
                try:
                    prompt_ids, attention_mask = self._left_pad_batch(prompt_ids_list, device)
                    out = _dllm_generate(
                        model,
                        prompt_ids,
                        self.tokenizer,
                        attention_mask=attention_mask,
                        gen_length=self.gen_length,
                        steps=self.diffusion_steps,
                        block_length=self.block_length,
                        temperature=0.0,
                        cfg_scale=0.0,
                        remasking="low_confidence",
                        mask_id=self.mask_id,
                    )
                    prompt_len = prompt_ids.shape[1]
                    for i, item in enumerate(batch):
                        prompt_text = self.tokenizer.decode(prompt_ids[i], skip_special_tokens=True)
                        generated = self.tokenizer.decode(out[i, prompt_len:], skip_special_tokens=False)
                        parsed = _extract_countdown_answer(generated)
                        score = _score_countdown(parsed, item["numbers"], item["target"])
                        is_correct = score == 1.0
                        if is_correct:
                            correct += 1
                        generation_rows.append(
                            {
                                "prompt": prompt_text,
                                "generation": generated,
                                "parsed_answer": parsed or "N/A",
                                "numbers": str(item["numbers"]),
                                "target": item["target"],
                                "correct": is_correct,
                            }
                        )
                except Exception as e:
                    print(f"[CountdownEvalCallback] generation error: {e}")

        correct_tensor = torch.tensor(correct, dtype=torch.long, device=device)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
        total_correct = correct_tensor.item()
        accuracy = total_correct / len(self.eval_items)

        if dist.is_available() and dist.is_initialized():
            all_rows_per_rank = [None] * world_size
            dist.all_gather_object(all_rows_per_rank, generation_rows)
            all_generation_rows = [row for rows in all_rows_per_rank for row in rows] if rank == 0 else None
        else:
            all_generation_rows = generation_rows

        if rank == 0:
            print(f"{'='*60}")
            print(
                f"[Step {state.global_step}] COUNTDOWN eval done: "
                f"{accuracy:.3f} ({total_correct}/{len(self.eval_items)})"
            )
            print(f"{'='*60}\n")

            n_print = min(3, len(all_generation_rows))
            for idx, row in enumerate(all_generation_rows[:n_print]):
                print(f"  [Gen {idx+1}] Numbers={row['numbers']} Target={row['target']} | Correct={row['correct']}")
                print(f"  {row['generation']}")
                print()

            if self.output_dir is not None:
                os.makedirs(self.output_dir, exist_ok=True)
                json_path = os.path.join(self.output_dir, f"eval_countdown_step_{state.global_step}.json")
                with open(json_path, "w") as f:
                    json.dump(
                        {"step": state.global_step, "accuracy": accuracy, "generations": all_generation_rows},
                        f,
                        indent=2,
                    )
                print(f"[CountdownEvalCallback] Saved {len(all_generation_rows)} examples to {json_path}")

            try:
                import wandb

                if wandb.run is not None:
                    table = wandb.Table(
                        columns=["prompt", "generation", "parsed_answer", "numbers", "target", "correct"]
                    )
                    for row in all_generation_rows:
                        table.add_data(
                            row["prompt"],
                            row["generation"],
                            row["parsed_answer"],
                            row["numbers"],
                            row["target"],
                            row["correct"],
                        )
                    wandb.log(
                        {
                            "eval/countdown_accuracy": accuracy,
                            "eval/countdown_generations": table,
                        }
                    )
                else:
                    print("[CountdownEvalCallback] wandb.run is None, skipping wandb log")
            except Exception as e:
                print(f"[CountdownEvalCallback] wandb log error: {e}")

        if was_training:
            model.train()


# ---------------------------------------------------------------------------
# ToolUse preprocessing and evaluation
# ---------------------------------------------------------------------------

# The dataset prompts contain a vague format description.  Replace it with one
# that uses <reasoning>/<answer> tags, consistent with all other tasks.
# The teacher conditioning reveals the <answer> span (Action + Action Input);
# the <reasoning> body remains masked during training.
_TOOLUSE_FORMAT_OLD = (
    "Use the following format:\n"
    "Thought: you should always think about what to do\n"
    "Action: the action to take, should be one of the tool names.\n"
    "Action Input: the input to the action, must be in JSON format."
    " All of the action input must be realistic and from the user."
)
_TOOLUSE_FORMAT_NEW = (
    "Respond in the following format:\n"
    "<reasoning>\n"
    "your step-by-step thinking about which tool to call and why\n"
    "</reasoning>\n"
    "<answer>\n"
    "Action: <exact tool name from the list above, nothing else>\n"
    "Action Input: <valid JSON object with the required parameters, nothing else>\n"
    "</answer>"
)


def _polish_tooluse_prompt(prompt: str) -> str:
    """Replace the vague format instruction with an explicit one."""
    return prompt.replace(_TOOLUSE_FORMAT_OLD, _TOOLUSE_FORMAT_NEW)


def preprocess_tooluse_for_self_distillation(
    data_path,
    tokenizer,
    max_length,
    min_reasoning_tokens=128,
    max_reasoning_tokens=256,
    test_split=0.01,
    max_prompt_length=None,
    cache_dir=None,
):
    """Preprocess tooluse JSONL (train.json) for self-distillation.

    Layout: [prompt][MASK * reasoning_len][answer_text][pad]
    where answer_text = "</reasoning>\\n<answer>\\nAction: <name>\\nAction Input: <json>\\n</answer>"

    The teacher sees the <answer> span revealed; the <reasoning> body remains
    masked.  mask_positions covers only the reasoning span.  No EOS appended —
    consistent with all other tasks (gsm8k, math, sciknoweval, code, etc.).
    """
    import json as _json

    items = []
    with open(data_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sample = _json.loads(line)
            question_text = _polish_tooluse_prompt(sample["prompt"])
            try:
                gt_list = _json.loads(sample["answer"])
                parts = []
                for item in gt_list:
                    parts.append(f"Action: {item['Action']}\nAction Input: {item['Action_Input']}")
                answer_text = "</reasoning>\n<answer>\n" + "\n".join(parts) + "\n</answer>"
            except (_json.JSONDecodeError, KeyError):
                continue
            items.append({"question_text": question_text, "answer_text": answer_text})

    random.shuffle(items)
    split = int(len(items) * test_split)
    train_items, test_items = items[split:], items[:split]

    train_data = _preprocess_masked_reasoning(
        train_items,
        tokenizer,
        max_length,
        min_reasoning_tokens,
        max_reasoning_tokens,
        desc="Preprocessing ToolUse train",
        max_prompt_length=max_prompt_length,
        cache_dir=cache_dir,
    )
    test_data = _preprocess_masked_reasoning(
        test_items,
        tokenizer,
        max_length,
        min_reasoning_tokens,
        max_reasoning_tokens,
        desc="Preprocessing ToolUse eval",
        max_prompt_length=max_prompt_length,
        cache_dir=cache_dir,
    )
    return train_data, test_data


def build_tooluse_eval_prompts(tokenizer, data_path, n=68, max_prompt_length=1024, seed=42):
    """Load examples from tooluse test.json for callback eval.

    Returns list of {"prompt_ids": Tensor[L], "ground_truth": str}
    where ground_truth is the raw answer JSON string from the dataset.
    """
    import json as _json

    items = []
    with open(data_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(_json.loads(line))

    indices = random.Random(seed).sample(range(len(items)), min(n, len(items)))
    eval_items = []
    for i in indices:
        sample = items[i]
        question_text = _polish_tooluse_prompt(sample["prompt"])
        prompt_msgs = [{"role": "user", "content": question_text}]
        prompt_text = tokenizer.apply_chat_template(prompt_msgs, tokenize=False) + "\n"
        prompt_ids = tokenizer(
            prompt_text, return_tensors="pt", truncation=True, max_length=max_prompt_length
        ).input_ids.squeeze(0)
        eval_items.append({"prompt_ids": prompt_ids, "ground_truth": sample["answer"]})
    return eval_items


class ToolUseEvalCallback(TrainerCallback):
    """
    Runs tooluse exact-match accuracy evaluation on the test split every
    `eval_steps` training steps.  Accuracy = fraction of examples where both
    Action names (bag) and Action Input values match ground truth exactly.
    Logs eval/tooluse_accuracy to wandb.
    """

    def __init__(
        self,
        eval_items,
        tokenizer,
        gen_length=256,
        block_length=32,
        eval_steps=50,
        eval_batch_size=4,
        output_dir=None,
        diffusion_steps=None,
        tokens_per_step=None,
        mask_id=126336,
        skip_first_step_eval=False,
    ):
        assert gen_length % block_length == 0, "gen_length must be divisible by block_length"
        self.eval_items = eval_items
        self.tokenizer = tokenizer
        self.gen_length = gen_length
        self.block_length = block_length
        self.eval_steps = eval_steps
        self.eval_batch_size = eval_batch_size
        self.output_dir = output_dir
        if diffusion_steps is not None:
            self.diffusion_steps = diffusion_steps
        elif tokens_per_step is not None:
            self.diffusion_steps = max(1, gen_length // tokens_per_step)
        else:
            self.diffusion_steps = gen_length
        self.mask_id = mask_id
        self.skip_first_step_eval = skip_first_step_eval

    def _left_pad_batch(self, prompt_ids_list, device):
        max_len = max(p.shape[0] for p in prompt_ids_list)
        pad_id = self.tokenizer.pad_token_id
        padded = torch.full((len(prompt_ids_list), max_len), pad_id, dtype=torch.long, device=device)
        attention_mask = torch.zeros(len(prompt_ids_list), max_len, dtype=torch.long, device=device)
        for i, p in enumerate(prompt_ids_list):
            padded[i, max_len - p.shape[0] :] = p.to(device)
            attention_mask[i, max_len - p.shape[0] :] = 1
        return padded, attention_mask

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if self.skip_first_step_eval:
            return
        self._run_eval(args, state, model)

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step % self.eval_steps != 0:
            return
        self._run_eval(args, state, model)

    def _run_eval(self, args, state, model):
        import json as _json
        import torch.distributed as dist
        from verl.utils.reward_score.feedback.tooluse import compute_score as _tooluse_score

        rank = _safe_rank()
        world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1

        device = next(model.parameters()).device
        was_training = model.training
        model.eval()

        if rank == 0:
            print(f"\n{'='*60}")
            print(
                f"[Step {state.global_step}] Running TOOLUSE eval on {len(self.eval_items)} examples..."
            )
            print(
                f"  gen_length={self.gen_length}, diffusion_steps={self.diffusion_steps}, "
                f"block_length={self.block_length}"
            )
            print(f"{'='*60}")

        my_items = self.eval_items[rank::world_size]
        correct = 0
        generation_rows = []
        with torch.no_grad():
            for batch_start in range(0, len(my_items), self.eval_batch_size):
                batch = my_items[batch_start : batch_start + self.eval_batch_size]
                prompt_ids_list = [item["prompt_ids"] for item in batch]
                try:
                    prompt_ids, attention_mask = self._left_pad_batch(prompt_ids_list, device)
                    out = _dllm_generate(
                        model,
                        prompt_ids,
                        self.tokenizer,
                        attention_mask=attention_mask,
                        gen_length=self.gen_length,
                        steps=self.diffusion_steps,
                        block_length=self.block_length,
                        temperature=0.0,
                        cfg_scale=0.0,
                        remasking="low_confidence",
                        mask_id=self.mask_id,
                    )
                    prompt_len = prompt_ids.shape[1]
                    for i, item in enumerate(batch):
                        prompt_text = self.tokenizer.decode(prompt_ids[i], skip_special_tokens=True)
                        generated = self.tokenizer.decode(out[i, prompt_len:], skip_special_tokens=False)
                        result = _tooluse_score(generated, item["ground_truth"])
                        is_correct = result["acc"] == 1.0
                        if is_correct:
                            correct += 1
                        generation_rows.append(
                            {
                                "prompt": prompt_text,
                                "generation": generated,
                                "parsed_answer": result["pred"],
                                "ground_truth": item["ground_truth"],
                                "correct": is_correct,
                                "feedback": result["feedback"],
                            }
                        )
                except Exception as e:
                    print(f"[ToolUseEvalCallback] generation error: {e}")

        correct_tensor = torch.tensor(correct, dtype=torch.long, device=device)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
        total_correct = correct_tensor.item()
        accuracy = total_correct / len(self.eval_items)

        if dist.is_available() and dist.is_initialized():
            all_rows_per_rank = [None] * world_size
            dist.all_gather_object(all_rows_per_rank, generation_rows)
            all_generation_rows = [row for rows in all_rows_per_rank for row in rows] if rank == 0 else None
        else:
            all_generation_rows = generation_rows

        if rank == 0:
            print(f"{'='*60}")
            print(
                f"[Step {state.global_step}] TOOLUSE eval done: "
                f"{accuracy:.3f} ({total_correct}/{len(self.eval_items)})"
            )
            print(f"{'='*60}\n")

            n_print = min(3, len(all_generation_rows))
            for idx, row in enumerate(all_generation_rows[:n_print]):
                print(f"  [Gen {idx+1}] GT={row['ground_truth']} | Correct={row['correct']}")
                print(f"  {row['generation']}")
                print()

            if self.output_dir is not None:
                os.makedirs(self.output_dir, exist_ok=True)
                json_path = os.path.join(self.output_dir, f"eval_tooluse_step_{state.global_step}.json")
                with open(json_path, "w") as f:
                    _json.dump(
                        {"step": state.global_step, "accuracy": accuracy, "generations": all_generation_rows},
                        f,
                        indent=2,
                    )
                print(f"[ToolUseEvalCallback] Saved {len(all_generation_rows)} examples to {json_path}")

            try:
                import wandb

                if wandb.run is not None:
                    table = wandb.Table(
                        columns=["prompt", "generation", "parsed_answer", "ground_truth", "correct", "feedback"]
                    )
                    for row in all_generation_rows:
                        table.add_data(
                            row["prompt"],
                            row["generation"],
                            row["parsed_answer"],
                            row["ground_truth"],
                            row["correct"],
                            row["feedback"],
                        )
                    wandb.log(
                        {
                            "eval/tooluse_accuracy": accuracy,
                            "eval/tooluse_generations": table,
                        }
                    )
                else:
                    print("[ToolUseEvalCallback] wandb.run is None, skipping wandb log")
            except Exception as e:
                print(f"[ToolUseEvalCallback] wandb log error: {e}")

        if was_training:
            model.train()
