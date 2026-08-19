"""
Self-Distillation Training Script for DLLM

This script trains the model using self-distillation:
- Forward pass 1: Prompt + fully masked response (unconditional)
- Forward pass 2: Prompt + response with answer revealed (conditional)
- Loss: KL divergence to match unconditional logits to conditional logits
"""

import torch
import argparse
from transformers import AutoTokenizer, AutoModel, TrainingArguments, BitsAndBytesConfig
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
import os
import random
import numpy as np

from self_distillation_trainer import (
    SelfDistillationTrainer,
    SelfDistillationDataCollator,
    SelfDistillationDataCollatorFrontReveal,
    SelfDistillationDataset,
    preprocess_gsm8k_for_self_distillation,
    preprocess_math500_for_self_distillation,
    preprocess_sciknoweval_for_self_distillation,
    preprocess_kodcode_for_self_distillation,
    preprocess_sudoku_for_self_distillation,
    preprocess_countdown_for_self_distillation,
    preprocess_tooluse_for_self_distillation,
    build_sciknoweval_eval_prompts,
    build_gsm8k_eval_prompts,
    build_math500_eval_prompts,
    build_mbpp_eval_prompts,
    build_humaneval_eval_prompts,
    build_sudoku_eval_prompts,
    build_countdown_eval_prompts,
    build_tooluse_eval_prompts,
    SciKnowEvalCallback,
    GSM8KEvalCallback,
    MATH500EvalCallback,
    CodeEvalCallback,
    HumanEvalEvalCallback,
    SudokuEvalCallback,
    CountdownEvalCallback,
    ToolUseEvalCallback,
    TrainingTimeCallback,
)


def init_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def parse_args():
    parser = argparse.ArgumentParser()

    # Hyperparameters
    parser.add_argument(
        "--model_name",
        type=str,
        default="/data0/shared/LLaDA-8B-Instruct/",
        help="Name of the pretrained model",
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for training")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument(
        "--max_prompt_length",
        type=int,
        default=None,
        help="Maximum prompt length (if set, prompt is truncated independently of max_length)",
    )
    parser.add_argument("--num_epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--grad_accum_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/data0/siyanzhao/",
        help="Directory to save model checkpoints",
    )
    parser.add_argument("--job_name", type=str, default="llada-self-distill", help="Job Name")
    parser.add_argument("--train_data", type=str, default="openai/gsm8k", help="Path to training data")
    parser.add_argument(
        "--dataset_type",
        type=str,
        default="gsm8k",
        choices=["gsm8k", "math500", "sciknoweval", "code", "sudoku", "countdown", "tooluse"],
        help="Dataset type: 'gsm8k', 'math500', 'sciknoweval', 'code', 'sudoku', or 'countdown'",
    )
    parser.add_argument(
        "--sciknoweval_domain",
        type=str,
        default=None,
        help="SciKnowEval domain filter (e.g. 'Biology', 'Chemistry'). None = all domains.",
    )
    parser.add_argument(
        "--min_reasoning_tokens",
        type=int,
        default=128,
        help="Min reasoning span length for SciKnowEval (sampled uniformly)",
    )
    parser.add_argument(
        "--max_reasoning_tokens",
        type=int,
        default=256,
        help="Max reasoning span length for SciKnowEval (sampled uniformly)",
    )
    parser.add_argument("--use_4bit", action="store_true", help="Load model in 4-bit NF4 quantization")
    parser.add_argument("--debugging", action="store_true", help="Disable wandb logging")
    parser.add_argument("--wandb_project", type=str, default="self-distillation", help="W&B project name")
    parser.add_argument(
        "--wandb_run_name", type=str, default=None, help="W&B run name (defaults to job_name)"
    )
    parser.add_argument(
        "--trainer_eval_steps", type=int, default=100, help="HuggingFace trainer eval frequency (steps)"
    )
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every N steps")
    parser.add_argument(
        "--callback_eval_steps", type=int, default=50, help="Accuracy callback eval frequency (steps)"
    )
    parser.add_argument(
        "--eval_n", type=int, default=50, help="Number of examples for callback evaluation set"
    )
    parser.add_argument(
        "--eval_batch_size", type=int, default=4, help="Batch size for callback accuracy evaluation"
    )
    parser.add_argument(
        "--eval_tokens_per_step",
        type=int,
        default=4,
        help="Tokens decoded per step during callback generation",
    )
    parser.add_argument(
        "--eval_diffusion_steps",
        type=int,
        default=None,
        help="Diffusion steps used by callback generation. If unset, uses max_reasoning_tokens // eval_tokens_per_step.",
    )
    parser.add_argument(
        "--eval_seed", type=int, default=42, help="Seed used to sample callback evaluation subset"
    )
    parser.add_argument(
        "--eval_completion_length",
        type=int,
        default=None,
        help="Generation length for callback eval. Defaults to max_length if not set.",
    )
    parser.add_argument(
        "--skip_first_step_eval",
        action="store_true",
        help="Skip the online eval that runs before training begins",
    )
    parser.add_argument(
        "--code_eval_datasets",
        nargs="+",
        choices=["mbpp", "humaneval"],
        default=["mbpp"],
        help="Which code eval datasets to run callbacks for (dataset_type=code only). E.g. --code_eval_datasets mbpp humaneval",
    )

    parser.add_argument(
        "--preprocess_cache_dir",
        type=str,
        default=os.path.join("/data1/siyanzhao", "preprocess_cache"),
        help="Directory to cache preprocessed datasets. Skips tokenization on subsequent runs.",
    )

    # Self-distillation specific
    parser.add_argument("--kl_weight", type=float, default=1.0, help="Weight for KL divergence loss")
    parser.add_argument(
        "--kl_type",
        type=str,
        default="forward",
        choices=["forward", "reverse", "jsd"],
        help="KL divergence type: 'forward' = KL(teacher||student), 'reverse' = KL(student||teacher)",
    )
    parser.add_argument(
        "--dynamic_teacher",
        action="store_true",
        help="If set, use the current student model (with LoRA) as teacher instead of the frozen base model",
    )
    parser.add_argument(
        "--ema_teacher",
        action="store_true",
        help="If set, use an EMA of the student's trainable weights as teacher",
    )
    parser.add_argument(
        "--ema_decay",
        type=float,
        default=0.999,
        help="EMA decay rate for the EMA teacher (default: 0.999)",
    )
    parser.add_argument(
        "--loss_type",
        type=str,
        default="kl",
        choices=["kl", "kl_ce", "ce"],
        help="Loss type: 'kl' = KL on reasoning only (default), 'kl_ce' = KL + CE on answer span, 'ce' = CE on answer span only",
    )
    parser.add_argument(
        "--ce_weight",
        type=float,
        default=1.0,
        help="Weight for CE loss on conditioning (answer) tokens (used when loss_type is kl_ce or ce)",
    )
    parser.add_argument(
        "--privileged_info_position",
        type=str,
        default="end",
        choices=["end", "front"],
        help=(
            "Where the ground-truth answer is revealed to the teacher. "
            "'end' (default) = trailing completion span (original SFCD). "
            "'front' = a sentence appended to the prompt, OPSD-style; completion is fully "
            "masked for both cond/uncond. Only implemented for --dataset_type sciknoweval."
        ),
    )

    return parser.parse_args()


def load_model_and_tokenizer(args):
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, padding_side="right", trust_remote_code=True, use_fast=True
    )
    tokenizer.pad_token = tokenizer.eos_token

    # Optionally load model in 4-bit NF4 quantization
    bnb_config = None
    if args.use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    model = AutoModel.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
    )

    # LoRA configuration
    lora_config = LoraConfig(
        r=128,
        lora_alpha=256,
        target_modules=["q_proj", "k_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model = model.to(torch.bfloat16)

    return tokenizer, model


def load_sciknoweval(domains=None, levels=None, types=None):
    from datasets import load_dataset as hf_load_dataset

    ds = hf_load_dataset("hicai-zju/SciKnowEval", split="test")
    if domains:
        ds = ds.filter(lambda x: x["domain"] in domains)
    if levels:
        ds = ds.filter(lambda x: x["details"]["level"] in levels)
    if types:
        ds = ds.filter(lambda x: x["type"] in types)
    return ds


def load_data(args, tokenizer):
    if args.dataset_type == "sciknoweval":
        domains = [args.sciknoweval_domain] if args.sciknoweval_domain else None
        data = load_sciknoweval(
            domains=domains,
            levels=["L3"],
            types=["mcq-4-choices", "mcq-2-choices"],
        )
        # SDPO-style split: 90/10 per domain, seed=42 (applied on raw data before tokenisation)
        splits = data.train_test_split(test_size=0.1, seed=42)
        raw_train, raw_test = splits["train"], splits["test"]
        train_data = preprocess_sciknoweval_for_self_distillation(
            raw_train,
            tokenizer,
            args.max_length,
            min_reasoning_tokens=args.min_reasoning_tokens,
            max_reasoning_tokens=args.max_reasoning_tokens,
            max_prompt_length=args.max_prompt_length,
            privileged_info_position=args.privileged_info_position,
        )
        eval_data = preprocess_sciknoweval_for_self_distillation(
            raw_test,
            tokenizer,
            args.max_length,
            min_reasoning_tokens=args.min_reasoning_tokens,
            max_reasoning_tokens=args.max_reasoning_tokens,
            max_prompt_length=args.max_prompt_length,
            privileged_info_position=args.privileged_info_position,
        )
        # Keep raw test split for the eval callback (needs original question/choices/answerKey)
        eval_items = build_sciknoweval_eval_prompts(raw_test, tokenizer, n=args.eval_n)
    elif args.dataset_type == "math500":
        data = load_dataset("ankner/math-500", split="train")
        train_data, eval_data = preprocess_math500_for_self_distillation(
            data,
            tokenizer,
            args.max_length,
            min_reasoning_tokens=args.min_reasoning_tokens,
            max_reasoning_tokens=args.max_reasoning_tokens,
        )
        eval_items = build_math500_eval_prompts(tokenizer, n=args.eval_n, seed=args.eval_seed)
    elif args.dataset_type == "sudoku":
        import pandas as pd
        from datasets import Dataset as HFDataset

        cur_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.normpath(os.path.join(cur_dir, "..", "dataset", "4x4_sudoku_unique_puzzles.csv"))
        df = pd.read_csv(csv_path, dtype={"Puzzle": str, "Solution": str})
        data = HFDataset.from_pandas(df)
        train_data, eval_data = preprocess_sudoku_for_self_distillation(
            data,
            tokenizer,
            args.max_length,
            min_reasoning_tokens=args.min_reasoning_tokens,
            max_reasoning_tokens=args.max_reasoning_tokens,
            cache_dir=args.preprocess_cache_dir,
        )
        eval_items = build_sudoku_eval_prompts(tokenizer, n=args.eval_n, seed=args.eval_seed)
    elif args.dataset_type == "countdown":
        data = load_dataset("Jiayi-Pan/Countdown-Tasks-3to4", split="train")
        data = data.filter(lambda x: len(x["nums"]) == 3)
        train_data, eval_data = preprocess_countdown_for_self_distillation(
            data,
            tokenizer,
            args.max_length,
            min_reasoning_tokens=args.min_reasoning_tokens,
            max_reasoning_tokens=args.max_reasoning_tokens,
            cache_dir=args.preprocess_cache_dir,
        )
        eval_items = build_countdown_eval_prompts(tokenizer, n=args.eval_n, seed=args.eval_seed)
    elif args.dataset_type == "tooluse":
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        tooluse_dir = os.path.join(cur_dir, "SDPO", "datasets", "tooluse")
        train_data, eval_data = preprocess_tooluse_for_self_distillation(
            data_path=os.path.join(tooluse_dir, "train.json"),
            tokenizer=tokenizer,
            max_length=args.max_length,
            min_reasoning_tokens=args.min_reasoning_tokens,
            max_reasoning_tokens=args.max_reasoning_tokens,
            max_prompt_length=args.max_prompt_length,
            cache_dir=args.preprocess_cache_dir,
        )
        eval_items = build_tooluse_eval_prompts(
            tokenizer=tokenizer,
            data_path=os.path.join(tooluse_dir, "test.json"),
            n=args.eval_n,
            max_prompt_length=args.max_prompt_length or 1024,
            seed=args.eval_seed,
        )
    elif args.dataset_type == "code":
        data = load_dataset("KodCode/KodCode-Light-RL-10K", split="train")
        train_data, eval_data = preprocess_kodcode_for_self_distillation(
            data,
            tokenizer,
            args.max_length,
            min_reasoning_tokens=args.min_reasoning_tokens,
            max_reasoning_tokens=args.max_reasoning_tokens,
        )
        # Build a dict of eval items per requested eval dataset
        eval_items = {}
        if "mbpp" in args.code_eval_datasets:
            eval_items["mbpp"] = build_mbpp_eval_prompts(tokenizer, n=args.eval_n, seed=args.eval_seed)
        if "humaneval" in args.code_eval_datasets:
            eval_items["humaneval"] = build_humaneval_eval_prompts(
                tokenizer, n=args.eval_n, seed=args.eval_seed
            )
    else:
        data = load_dataset(args.train_data, "main", split="train")
        # Keep training data on train split, but align callback accuracy to GSM8K test split.
        callback_eval_data = load_dataset(args.train_data, "main", split="test")
        train_data, eval_data = preprocess_gsm8k_for_self_distillation(
            data,
            tokenizer,
            args.max_length,
            min_reasoning_tokens=args.min_reasoning_tokens,
            max_reasoning_tokens=args.max_reasoning_tokens,
        )
        eval_items = build_gsm8k_eval_prompts(
            callback_eval_data,
            tokenizer,
            n=args.eval_n,
            seed=args.eval_seed,
        )

    print(f"Train data length: {len(train_data)}")
    print(f"Eval data length: {len(eval_data)}")

    train_dataset = SelfDistillationDataset(train_data, tokenizer, args.max_length)
    eval_dataset = SelfDistillationDataset(eval_data, tokenizer, args.max_length)

    return train_dataset, eval_dataset, eval_items


def train_model(args, tokenizer, model):
    # Load dataset
    train_dataset, eval_dataset, eval_items = load_data(args, tokenizer)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=os.path.join(args.output_dir, args.job_name),
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        eval_strategy="steps",
        eval_steps=args.trainer_eval_steps,
        logging_steps=2,
        save_steps=args.save_steps,
        save_total_limit=20,
        learning_rate=args.learning_rate,
        load_best_model_at_end=True,
        weight_decay=0.1,
        max_grad_norm=1.0,
        bf16=True,
        report_to="wandb" if not args.debugging else "none",
        run_name=args.wandb_run_name or args.job_name,
        remove_unused_columns=False,
    )

    # Build callback list — TrainingTimeCallback must be first so its on_step_end
    # fires before eval callbacks, capturing only training compute time.
    callbacks = [TrainingTimeCallback()]
    if eval_items is not None:
        block_length = 32
        _gen_len_base = (
            args.eval_completion_length if args.eval_completion_length is not None else args.max_length
        )
        gen_length = (_gen_len_base // block_length) * block_length  # snap to multiple of 32
        eval_diffusion_steps = args.eval_diffusion_steps
        if eval_diffusion_steps is None:
            eval_diffusion_steps = gen_length // args.eval_tokens_per_step
        mask_id = tokenizer.mask_token_id if tokenizer.mask_token_id is not None else 126336
        if args.dataset_type == "math500":
            callbacks.append(
                MATH500EvalCallback(
                    eval_items=eval_items,
                    tokenizer=tokenizer,
                    gen_length=gen_length,
                    block_length=block_length,
                    eval_steps=args.callback_eval_steps,
                    eval_batch_size=args.eval_batch_size,
                    output_dir=os.path.join(args.output_dir, args.job_name, "eval_generations"),
                    diffusion_steps=eval_diffusion_steps,
                    mask_id=mask_id,
                    skip_first_step_eval=args.skip_first_step_eval,
                )
            )
        elif args.dataset_type == "sciknoweval":
            callbacks.append(
                SciKnowEvalCallback(
                    eval_items=eval_items,
                    tokenizer=tokenizer,
                    gen_length=gen_length,
                    block_length=block_length,
                    eval_steps=args.callback_eval_steps,
                    eval_batch_size=args.eval_batch_size,
                    output_dir=os.path.join(args.output_dir, args.job_name, "eval_generations"),
                    diffusion_steps=eval_diffusion_steps,
                    mask_id=mask_id,
                    skip_first_step_eval=args.skip_first_step_eval,
                )
            )
        elif args.dataset_type == "sudoku":
            callbacks.append(
                SudokuEvalCallback(
                    eval_items=eval_items,
                    tokenizer=tokenizer,
                    gen_length=gen_length,
                    block_length=block_length,
                    eval_steps=args.callback_eval_steps,
                    eval_batch_size=args.eval_batch_size,
                    output_dir=os.path.join(args.output_dir, args.job_name, "eval_generations"),
                    diffusion_steps=eval_diffusion_steps,
                    mask_id=mask_id,
                    skip_first_step_eval=args.skip_first_step_eval,
                )
            )
        elif args.dataset_type == "countdown":
            callbacks.append(
                CountdownEvalCallback(
                    eval_items=eval_items,
                    tokenizer=tokenizer,
                    gen_length=gen_length,
                    block_length=block_length,
                    eval_steps=args.callback_eval_steps,
                    eval_batch_size=args.eval_batch_size,
                    output_dir=os.path.join(args.output_dir, args.job_name, "eval_generations"),
                    diffusion_steps=eval_diffusion_steps,
                    mask_id=mask_id,
                    skip_first_step_eval=args.skip_first_step_eval,
                )
            )
        elif args.dataset_type == "tooluse":
            callbacks.append(
                ToolUseEvalCallback(
                    eval_items=eval_items,
                    tokenizer=tokenizer,
                    gen_length=gen_length,
                    block_length=block_length,
                    eval_steps=args.callback_eval_steps,
                    eval_batch_size=args.eval_batch_size,
                    output_dir=os.path.join(args.output_dir, args.job_name, "eval_generations"),
                    diffusion_steps=eval_diffusion_steps,
                    mask_id=mask_id,
                    skip_first_step_eval=args.skip_first_step_eval,
                )
            )
        elif args.dataset_type == "code":
            _callback_cls = {"mbpp": CodeEvalCallback, "humaneval": HumanEvalEvalCallback}
            for dataset_name, items in eval_items.items():
                callbacks.append(
                    _callback_cls[dataset_name](
                        eval_items=items,
                        tokenizer=tokenizer,
                        gen_length=gen_length,
                        block_length=block_length,
                        eval_steps=args.callback_eval_steps,
                        eval_batch_size=args.eval_batch_size,
                        output_dir=os.path.join(args.output_dir, args.job_name, "eval_generations"),
                        diffusion_steps=eval_diffusion_steps,
                        mask_id=mask_id,
                        skip_first_step_eval=args.skip_first_step_eval,
                    )
                )
        else:
            callbacks.append(
                GSM8KEvalCallback(
                    eval_items=eval_items,
                    tokenizer=tokenizer,
                    gen_length=gen_length,
                    block_length=block_length,
                    eval_steps=args.callback_eval_steps,
                    eval_batch_size=args.eval_batch_size,
                    output_dir=os.path.join(args.output_dir, args.job_name, "eval_generations"),
                    diffusion_steps=eval_diffusion_steps,
                    mask_id=mask_id,
                    skip_first_step_eval=args.skip_first_step_eval,
                )
            )

    # Initialize trainer
    collator_cls = (
        SelfDistillationDataCollatorFrontReveal
        if args.privileged_info_position == "front"
        else SelfDistillationDataCollator
    )
    trainer = SelfDistillationTrainer(
        model=model,
        args=training_args,
        data_collator=collator_cls(tokenizer=tokenizer, mask_token_id=126336),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        kl_weight=args.kl_weight,
        kl_type=args.kl_type,
        fixed_teacher=not args.dynamic_teacher and not args.ema_teacher,
        ema_teacher=args.ema_teacher,
        ema_decay=args.ema_decay,
        loss_type=args.loss_type,
        ce_weight=args.ce_weight,
        callbacks=callbacks if callbacks else None,
    )

    # Auto-detect latest checkpoint and resume if one exists
    checkpoint_dir = os.path.join(args.output_dir, args.job_name)
    checkpoints = (
        sorted(
            [d for d in os.listdir(checkpoint_dir) if d.startswith("checkpoint-")],
            key=lambda x: int(x.split("-")[1]),
        )
        if os.path.isdir(checkpoint_dir)
        else []
    )
    resume_from = os.path.join(checkpoint_dir, checkpoints[-1]) if checkpoints else None
    if resume_from:
        print(f"Resuming from checkpoint: {resume_from}")

    # Start training
    trainer.train(resume_from_checkpoint=resume_from)


if __name__ == "__main__":
    init_seed(42)
    args = parse_args()
    os.environ["WANDB_PROJECT"] = args.wandb_project
    tokenizer, model = load_model_and_tokenizer(args)
    train_model(args, tokenizer, model)
