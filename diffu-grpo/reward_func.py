import numpy as np
import re
from math500_utils import remove_boxed, last_boxed_only_string, is_equiv, boxed_in_answer
import multiprocessing
import resource
import os

import time
import random
import string
import shutil


def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()


def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()


def correctness_reward_func(prompts, completions, answer, step=None, run_name=None, **kwargs) -> list[float]:
    responses = [completion[0]["content"] for completion in completions]
    q = prompts[0][-1]["content"]
    extracted_responses = [extract_xml_answer(r) for r in responses]

    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    RESET = "\033[0m"

    print(
        "-" * 20,
        f"\n{RED}Prompt:{RESET}\n{q}\n",
        "-" * 20,
        f"\n{GREEN}Ground Truth:{RESET}\n{answer[0]}\n",
        "-" * 20,
        f"\n{BLUE}Response:{RESET}\n{responses[0]}\n",
        "-" * 20,
        f"\n{YELLOW}Extracted:{RESET}\n{extracted_responses[0]}\n",
    )
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]


def int_reward_func(completions, **kwargs) -> list[float]:
    responses = [completion[0]["content"] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]


def strict_format_reward_func(completions, **kwargs) -> list[float]:
    # functions adopted from Will Brown's GRPO demo:
    # https://gist.github.com/willccbb/4676755236bb08cab5f4e54a0475d6fb
    # re.DOTALL allows `.` to match newline characters inside multi-line reasoning.
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r, flags=re.DOTALL) for r in responses]
    return [0.5 if match else 0.0 for match in matches]


def soft_format_reward_func(completions, **kwargs) -> list[float]:
    # functions adopted from Will Brown's GRPO demo:
    # https://gist.github.com/willccbb/4676755236bb08cab5f4e54a0475d6fb
    # re.DOTALL allows `.` to match newline characters inside multi-line reasoning.
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r, flags=re.DOTALL) for r in responses]
    return [0.5 if match else 0.0 for match in matches]


def count_xml(text) -> float:
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1]) * 0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1) * 0.001
    return count


def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]


def reward_len(completions, **kwargs):
    # run this reward function for sanity check
    # return [abs(5 - len(completion[0]["content"])) for completion in completions]
    return [-len(completion[0]["content"]) for completion in completions]


def extract_solution(solution_str):
    answer_pattern = r"<answer>(.*?)</answer>"
    matches = re.findall(answer_pattern, solution_str, re.DOTALL)
    return matches[-1].strip() if matches else None


def validate_equation(equation_str, available_numbers):
    """Validate that equation only uses available numbers and each number once."""
    try:
        numbers_in_eq = [int(n) for n in re.findall(r"\d+", equation_str)]
        return sorted(numbers_in_eq) == sorted(available_numbers)
    except:
        return False


def evaluate_equation(equation_str):
    try:
        allowed_pattern = r"^[\d+\-*/().\s]+$"
        if not re.match(allowed_pattern, equation_str):
            raise ValueError("Invalid characters in equation.")
        return eval(equation_str, {"__builtins__": None}, {})
    except:
        return None


def compute_score(solution_str, ground_truth, method="strict", format_score=0.1, score=1.0):
    target = ground_truth["target"]
    numbers = ground_truth["numbers"]

    equation = extract_solution(solution_str)
    do_print = np.random.rand() < 0.4

    if do_print:
        print(f"--------------------------------")
        print(f"Target: {target} | Numbers: {numbers}")
        print(f"Extracted equation: {equation}")
        print(f"Solution string: {solution_str}")

    if equation is None:
        if do_print:
            print(f"No equation found")
        return 0

    if not validate_equation(equation, numbers):
        if do_print:
            print(f"Invalid equation")
        return format_score

    try:
        result = evaluate_equation(equation)
        if result is None:
            if do_print:
                print(f"Could not evaluate equation")
            return format_score

        if abs(result - target) < 1e-5:
            if do_print:
                print(f"Correct equation: {equation} = {result}")
            return score
        else:
            if do_print:
                print(f"Wrong result: equation = {result}, target = {target}")
            return format_score
    except:
        if do_print:
            print(f"Error evaluating equation")
        return format_score


def countdown_reward_func(prompts, completions, run_name, step=None, rank=None, **kwargs) -> list[float]:
    if (
        isinstance(completions[0], list)
        and isinstance(completions[0][0], dict)
        and "content" in completions[0][0]
    ):
        responses = [completion[0]["content"] for completion in completions]
    else:
        responses = completions

    scores = []
    for i, response in enumerate(responses):
        ground_truth = {"target": kwargs["target"][i], "numbers": kwargs["numbers"][i]}
        scores.append(compute_score(response, ground_truth))

    return scores


def extract_answer_sudoku(solution_str):
    answer_pattern = r"<answer>(.*?)</answer>"
    matches = re.findall(answer_pattern, solution_str, re.DOTALL)
    if matches:
        return "".join(char for char in matches[-1].strip() if char.isdigit())
    return None


def validate_sudoku_solution(solution_str, ground_truth, puzzle):
    if solution_str is None or len(solution_str) == 0:
        return 0.0

    if len(solution_str) < 16:
        # Pad with zeros if too short
        solution_str = solution_str + "0" * (16 - len(solution_str))
    elif len(solution_str) > 16:
        # Truncate if too long
        solution_str = solution_str[:16]

    empty_indices = [i for i in range(16) if puzzle[i] == "0"]

    if empty_indices:
        correct_cells = sum(1 for i in empty_indices if solution_str[i] == ground_truth[i])
        return correct_cells / len(empty_indices)
    return 0.0


def sudoku_reward_func(prompts, completions, run_name, step=None, rank=None, **kwargs) -> list[float]:
    if (
        isinstance(completions[0], list)
        and isinstance(completions[0][0], dict)
        and "content" in completions[0][0]
    ):
        responses = [completion[0]["content"] for completion in completions]
    else:
        responses = completions

    scores = []
    for i, response in enumerate(responses):
        do_print = np.random.rand() < 0.4
        puzzle = kwargs["puzzle"][i]
        ground_truth = kwargs["solution"][i]
        solution = extract_answer_sudoku(response)

        score = 0.0 if solution is None else validate_sudoku_solution(solution, ground_truth, puzzle)
        scores.append(score)

        if do_print:
            print(f"--------------------------------")
            print(f"Puzzle: {puzzle} (length: {len(puzzle)})")
            print(f"Extracted solution: {solution}  (length: {len(solution) if solution else 0})")
            print(f"Ground_truth: {ground_truth}")
            print(f"Score: {score:.4f}")

    return scores


def correctness_reward_func_math(
    prompts, completions, answer, step=None, run_name=None, **kwargs
) -> list[float]:
    boxed_in_answer_rewards = boxed_in_answer(prompts, completions, answer, step=step)
    responses = [completion[0]["content"] for completion in completions]
    q = prompts[0][-1]["content"]
    extracted_responses = []
    answer = [remove_boxed(last_boxed_only_string(a)) for a in answer]
    for r in responses:
        try:
            r = remove_boxed(last_boxed_only_string(r))
        except:
            pass
        extracted_responses.append(r)
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    RESET = "\033[0m"

    print(
        "-" * 20,
        f"\n{RED}Question:{RESET}\n{q}",
        "-" * 20,
        f"\n{GREEN}Ground Truth:{RESET}\n{answer[0]}",
        "-" * 20,
        f"\n{BLUE}Response:{RESET}\n{responses[0]}",
        "-" * 20,
        f"\n{YELLOW}Extracted:{RESET}\n{extracted_responses[0]}",
    )
    print("✅" if is_equiv(extracted_responses[0], answer[0]) else "❌")

    return [2.0 if is_equiv(r, a) else 0.0 for r, a in zip(extracted_responses, answer)]


def boxed_and_answer_tags_format_reward(
    prompts, completions, answer, step=None, run_name=None, **kwargs
) -> list[float]:
    boxed_in_answer_rewards = boxed_in_answer(prompts, completions, answer, step=step)
    rewards = [b * 0.5 for b in boxed_in_answer_rewards]
    return rewards


def run_test(test_func_name, code_str, result_dict, cwd_path, rank):
    cwd_path = cwd_path + "/" + str(rank)
    os.makedirs(cwd_path, exist_ok=True)

    def target():
        try:
            # Set memory limit to 1 GB (in bytes)
            soft, hard = 1_000_000_000, 1_000_000_000
            resource.setrlimit(resource.RLIMIT_AS, (soft, hard))

            # Change working directory
            os.chdir(cwd_path)

            # Create a new namespace and execute code
            local_ns = {}
            exec(code_str, local_ns)
            local_ns[test_func_name]()
            result_dict[test_func_name] = True
        except Exception:
            result_dict[test_func_name] = False

    proc = multiprocessing.Process(target=target)
    proc.start()
    proc.join(timeout=1.0)

    if proc.is_alive():
        proc.terminate()
        result_dict[test_func_name] = False


def split_test_function(test_code: str, base_name: str = "test_case"):
    lines = test_code.strip().splitlines()
    result = []
    counter = 1

    for line in lines:
        line = line.strip()
        if line.startswith("assert"):
            fn = f"def {base_name}_{counter}():\n    {line}"
            result.append(fn)
            counter += 1

    return "\n\n".join(result)


BLOCKED_MODULES = [
    " os",
    " sys",
    " subprocess",
    " shutil",
    " socket",
    " psutil",
    " ctypes",
    " pathlib",
    " builtins",
    "__import__",
]


def is_safe_code(code_str):
    """Reject code that contains any blocked modules or keywords."""
    for blocked in BLOCKED_MODULES:
        if blocked in code_str:
            return False
    return True


def time_based_random_string(length=10):
    seed = int(time.time() * 1e6)
    random.seed(seed)
    return "".join(random.choices(string.ascii_letters + string.digits, k=length))


def coding_reward_func(prompts, completions, answer, step=None, run_name=None, **kwargs) -> list[float]:
    execution_cwd = kwargs.get("cwd_path")
    if not os.path.exists(execution_cwd):
        os.makedirs(execution_cwd, exist_ok=True)
    programs = []
    for group in completions:
        for message in group:
            content = message["content"]

            # Look for the first python code block
            code_match = re.search(r"```python\n(.*?)```", content, re.DOTALL)
            code = code_match.group(1) if code_match else ""

            # Check if the code is within <answer>...</answer>
            answer_match = re.search(r"<answer>(.*?)</answer>", content, re.DOTALL)
            is_in_answer = False
            if answer_match and code:
                answer_content = answer_match.group(1)
                is_in_answer = f"```python\n{code}```" in answer_content

            programs.append((code, is_in_answer))

    unit_tests = [entry["tests"] for entry in answer]
    rewards = []
    for i, (solution, tests) in enumerate(zip(programs, unit_tests)):
        solution, is_in_answer = solution
        is_in_answer_reward = 0.5 if is_in_answer else 0
        # Step 1: Extract imported function name
        import_match = re.search(r"from solution import (\w+)", tests)
        if not import_match:
            assert_match = re.search(r"assert\s+(\w+)\s*\(", tests)
            if assert_match:
                imported_func = assert_match.group(1)
            else:
                print("No import or assert-based function name found in test code.")
                print("=" * 10, "tests", "=" * 10)
                print(tests)
                print("=" * 30)
                rewards.append(0)
                continue
        else:
            imported_func = import_match.group(1)

        # Step 2: Extract defined function name from solution
        solution_match = re.search(r"def (\w+)\(", solution)
        if not solution_match:
            print("No function definition in generation")
            print("=" * 10, "model output", "=" * 10)
            print(solution)
            print("=" * 10)
            rewards.append(0 + is_in_answer_reward)
            continue
        defined_func = solution_match.group(1)

        # Step 3: Rename if function names differ
        if defined_func != imported_func:
            solution = re.sub(rf"\bdef {defined_func}\b", f"def {imported_func}", solution)

        # Step 3.5: Check iff code is safe to run
        if not is_safe_code(solution):
            rewards.append(0 + is_in_answer_reward)
            print(f"Potentially Unsafe Generation:\n{solution}")
            continue
        # Step 4: Extract test function names
        test_funcs = re.findall(r"def (\w+)\(\):", tests)

        if len(test_funcs) <= 1:
            tests = split_test_function(tests)
            print(f"Fixed Test functions\n{tests}")
            test_funcs = re.findall(r"def (\w+)\(\):", tests)

        # Step 5: Replace import with the solution code
        if import_match:
            test_code = re.sub(r"from solution import \w+", lambda _: solution, tests)
        elif assert_match:
            test_code = solution + tests

        # Step 6: Build complete executable test code
        # Dict to hold test results
        manager = multiprocessing.Manager()
        result_dict = manager.dict()

        # Run all test functions in parallel
        jobs = []
        for rank, fn in enumerate(test_funcs):
            p = multiprocessing.Process(
                target=run_test, args=(fn, test_code, result_dict, execution_cwd, rank)
            )
            p.start()
            jobs.append(p)

        for p in jobs:
            p.join()

        # Compute reward
        passed = sum(result_dict.get(fn, False) for fn in test_funcs)
        total = len(test_funcs)
        reward = passed / total if total > 0 else 0.0
        rewards.append(reward + is_in_answer_reward)
    return rewards
