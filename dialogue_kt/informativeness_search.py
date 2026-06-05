"""
Stepwise Informativeness Search utilities for Dialogue-KT.

Adapted from Wang et al. (2025), "Stepwise Informativeness Search for
Effective and Efficient LLM Reasoning". The paper reduces reasoning errors by
prioritizing underutilized prior steps and filtering redundant conclusions. For
Dialogue-KT, dialogue turns are treated as reasoning/evidence steps: recent
turns and non-redundant prior turns are marked as informative sources for the
current KC prediction.
"""

from __future__ import annotations

from collections import Counter
from typing import Iterable, List

import numpy as np


MARKER_INFO = "<informative_step>"
MARKER_RECENT = "<recent_step>"
MARKER_END = "</step>"


def _trigrams(tokens: List[str]) -> Counter:
    if len(tokens) < 3:
        return Counter([" ".join(tokens)]) if tokens else Counter()
    return Counter(" ".join(tokens[i:i + 3]) for i in range(len(tokens) - 2))


def _tokenize(text: str) -> List[str]:
    return text.lower().replace("\n", " ").split()


def trigram_overlap(a: str, b: str) -> float:
    """Return overlap of a's trigrams covered by b, matching the paper's heuristic."""
    a_tri = _trigrams(_tokenize(a))
    if not a_tri:
        return 0.0
    b_tri = _trigrams(_tokenize(b))
    return sum((a_tri & b_tri).values()) / max(1, sum(a_tri.values()))


def turn_conclusion(turn: dict) -> str:
    """Use the student's response as the turn conclusion when available."""
    student = (turn.get("student") or "").strip()
    if student:
        return student
    return (turn.get("teacher") or "").strip()


def turn_full_text(turn: dict) -> str:
    parts = []
    if turn.get("teacher"):
        parts.append(str(turn["teacher"]))
    if turn.get("student"):
        parts.append(str(turn["student"]))
    return " ".join(parts)


def informativeness_strengths(
    dialogue: List[dict],
    turn_idx: int,
    mode: str = "novelty",
    threshold: float = 0.7,
) -> np.ndarray:
    """Return per-turn informativeness strengths.

    Values:
    - 0: unmarked background
    - 1: recent/local step
    - 2: informative underutilized/non-redundant step

    We compute information gain for visible prior turns as
    1 - max trigram overlap between that turn's conclusion and later visible
    turns. Low-overlap turns are underutilized/non-redundant and are marked.
    """
    base_mode = mode.replace("_prune", "")
    if base_mode == "recent":
        base_mode = "recent_grounded"

    strengths = np.zeros(len(dialogue), dtype=np.int64)
    if not dialogue:
        return strengths

    cur = max(0, min(len(dialogue) - 1, turn_idx - 1 if turn_idx > 0 else turn_idx))
    visible = list(range(cur + 1))

    # Always keep local recency available.
    strengths[cur] = max(strengths[cur], 1)
    if cur >= 1:
        strengths[cur - 1] = max(strengths[cur - 1], 1)

    if base_mode == "recent_grounded":
        strengths[max(0, cur - 2):cur + 1] = 2
        return strengths

    conclusions = [turn_conclusion(dialogue[i]) for i in visible]
    for pos, idx in enumerate(visible[:-1]):
        cur_conc = conclusions[pos]
        if len(_tokenize(cur_conc)) < 5:
            continue
        later = conclusions[pos + 1:]
        max_sim = max((trigram_overlap(cur_conc, other) for other in later), default=0.0)
        info_gain = 1.0 - max_sim
        if info_gain >= threshold:
            strengths[idx] = 2

    # Current question is the required target step; recent previous evidence is useful.
    strengths[cur] = 2
    if base_mode == "novelty_recent" and cur >= 1:
        strengths[cur - 1] = 2
    return strengths


def wrap_step(line: str, strength: int) -> str:
    if strength >= 2:
        return f"{MARKER_INFO} {line} {MARKER_END}"
    if strength == 1:
        return f"{MARKER_RECENT} {line} {MARKER_END}"
    return line


def informativeness_system_instruction() -> str:
    return (
        "Treat dialogue turns as reasoning/evidence steps. "
        f"{MARKER_INFO} marks non-redundant or underutilized steps whose information should be explicitly grounded in the current KC judgment. "
        f"{MARKER_RECENT} marks local recent context. Prefer informative non-redundant evidence over repeated or irrelevant turns, "
        "and avoid circularly relying on the same conclusion multiple times."
    )
