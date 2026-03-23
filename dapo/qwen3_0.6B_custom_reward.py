import re
from typing import Any, Optional

from verl.utils.reward_score import gsm8k as gsm8k_score

def check_thinking_format(solution_str):
    pattern = r"<think>.*?</think>"
    matches = re.findall(pattern, solution_str, re.DOTALL)
    # return match is not None
    return matches is not None and len(matches)

def check_assitant_end(solution_str):
    return solution_str.rstrip().endswith("</assistant>")

def _compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info = None,
    *,
    method = "strict",
    format_score = 0.0,
    correct_score = 1.0,
    format_bonus = 0.0,
    check_thinking = True,
    check_last_assitant = True,
    last_assitant_bonus = 0.1,
    **kwargs,
):
    base_score = gsm8k_score.compute_score(
        solution_str=solution_str,
        ground_truth=ground_truth,
        method=method,
        format_score=format_score,
        score=correct_score,
    )

    if base_score == 0.0:
        base_score = -0.3

    bonus = 0.0

    if check_thinking:
        if check_thinking_format(solution_str):
            bonus += 0.2
        else:
            bonus -= 0.2

    return float(base_score + bonus)

def compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info = None,
    **kwargs,
):
    return _compute_score(
        data_source=data_source,
        solution_str=solution_str,
        ground_truth=ground_truth,
        extra_info=extra_info,
        **kwargs,
    )
