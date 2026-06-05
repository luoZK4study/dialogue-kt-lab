"""
Bayesian Epistemology utilities for Dialogue-KT.

Inspired by Kim et al. (2025), "From Evidence to Belief: A Bayesian
Epistemology Approach to Language Models". The paper studies how LLM
confidence changes with evidence informativeness and reliability. For
Dialogue-KT, we adapt this as prompt-side evidence reliability guidance:
recent direct student responses and prior correctness labels are treated as
strong evidence, older/indirect context as weaker evidence, and conflicting
signals should reduce confidence.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np


MARKER_STRONG = "<strong_evidence>"
MARKER_WEAK = "<weak_evidence>"
MARKER_BACKGROUND = "<background_evidence>"
MARKER_END = "</evidence>"


def bayes_evidence_mask(n_turns: int, turn_idx: int, mode: str = "adaptive") -> np.ndarray:
    """Return per-turn evidence strength for Bayesian evidence prompts.

    Values:
    - 0: unmarked/background dialogue context
    - 1: weak evidence
    - 2: strong evidence

    The adaptive policy follows the paper's reliability/informativeness idea:
    the current teacher question and the most recent student attempts are most
    informative for the current belief, while older turns are useful but weaker.
    """
    strengths = np.zeros(n_turns, dtype=np.int64)
    if n_turns == 0:
        return strengths

    # turn_idx is the dialogue turn number, generally 1-indexed except rare
    # MathDial student-initiated turn 0. Convert to list index conservatively.
    cur = max(0, min(n_turns - 1, turn_idx - 1 if turn_idx > 0 else turn_idx))

    if mode == "recent2":
        start = max(0, cur - 1)
        strengths[start:cur + 1] = 2
    elif mode == "recent3":
        start = max(0, cur - 2)
        strengths[start:cur + 1] = 2
    else:  # adaptive
        # Need enough history before marking older evidence; for short contexts,
        # mark only the current turn to avoid over-trusting sparse evidence.
        strengths[cur] = 2
        if cur >= 1:
            strengths[cur - 1] = 2
        if cur >= 3:
            strengths[cur - 2] = 1
        if cur >= 5:
            strengths[max(0, cur - 4):cur - 2] = 1

    return strengths


def wrap_evidence(line: str, strength: int) -> str:
    if strength >= 2:
        return f"{MARKER_STRONG} {line} {MARKER_END}"
    if strength == 1:
        return f"{MARKER_WEAK} {line} {MARKER_END}"
    return line


def bayesian_system_instruction(include_labels: bool = False) -> str:
    label_note = (
        " Treat previous correctness and knowledge-component labels as high-reliability observed evidence, "
        "but never infer the current answer from labels that are not in the past."
        if include_labels else ""
    )
    return (
        f"Within the dialogue, {MARKER_STRONG} marks highly informative and reliable evidence, "
        f"{MARKER_WEAK} marks useful but less direct evidence, and unmarked turns may be background or irrelevant. "
        "Update your belief about the student's current knowledge component like Bayesian evidence updating: "
        "increase confidence when reliable evidence supports mastery, decrease confidence when reliable evidence supports a misconception, "
        "and avoid overreacting to weak, coincidental, old, or irrelevant evidence."
        + label_note
    )
