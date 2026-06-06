"""
Information Bottleneck inspired methods for Dialogue-KT.

IBCircuit (Bian 2025): Uses IB principle to identify the most informative
model components. Adapted here to identify the most informative dialogue
turns for KC mastery prediction.

Key adaptation: Instead of finding circuits in model components, we find
"evidence circuits" in dialogue turns — the minimal set of turns that
retain predictive information about KC mastery.
"""
from typing import List, Optional
import numpy as np
import re
import math
from collections import Counter

from dialogue_kt.prompting import get_dialogue_text, get_mathdial_context, correct_to_str, standards_to_str


def method_system_instruction_ib_turns(args):
    return (
        "Focus on the most informative prior dialogue turns marked with "
        "<informative_turn> tags. These turns have been selected because "
        "they carry the most predictive information about the student's "
        "knowledge state. Less informative turns are shown without markers."
    )


def _tokenize(text: str):
    return set(re.findall(r"[a-zA-Z0-9]+", (text or "").lower()))


def _compute_turn_informativeness(
    dialogue: List[dict],
    turn_idx: int,
    kc: Optional[str],
    alpha: float = 0.4,  # KC relevance weight
    beta: float = 0.3,   # Surprise (incorrect=more informative) weight
    gamma: float = 0.3,  # Lexical novelty weight
):
    """
    Compute IB-inspired informativeness score for each historical turn.

    The score balances:
    1. KC relevance: How related is this turn to the target KC?
    2. Surprise: Incorrect answers carry more information (IB principle)
    3. Novelty: How unique is this turn's content vs. other turns?
    """
    prior_turns = [t for t in dialogue if t["turn"] < turn_idx and t.get("correct") is not None]
    if not prior_turns:
        return np.array([]), []

    n = len(prior_turns)
    scores = np.zeros(n)

    # Gather all turn texts for novelty computation
    turn_texts = []
    for t in prior_turns:
        tt = f"{t.get('teacher', '')} {t.get('student', '')} {' '.join(t.get('kcs', []))}"
        turn_texts.append(tt)

    kc_tokens = _tokenize(kc or "")

    for i, (t, text) in enumerate(zip(prior_turns, turn_texts)):
        # 1. KC relevance: Jaccard similarity between turn KCs and target KC
        turn_kc_text = " ".join(t.get("kcs", []))
        turn_kc_tokens = _tokenize(turn_kc_text)
        kc_rel = len(kc_tokens & turn_kc_tokens) / max(1, len(kc_tokens | turn_kc_tokens))

        # 2. Surprise: incorrect turns carry more information (higher entropy)
        surprise = 0.7 if t.get("correct") is False else 0.3

        # 3. Lexical novelty: how different is this turn from others?
        turn_tokens = _tokenize(text)
        max_overlap = 0.0
        for j, other_text in enumerate(turn_texts):
            if i == j:
                continue
            other_tokens = _tokenize(other_text)
            overlap = len(turn_tokens & other_tokens) / max(1, len(turn_tokens | other_tokens))
            max_overlap = max(max_overlap, overlap)
        novelty = 1.0 - max_overlap  # High novelty = low overlap with any other turn

        # Combined IB score
        scores[i] = alpha * kc_rel + beta * surprise + gamma * novelty

    # Normalize scores to [0, 1]
    if scores.max() > scores.min():
        scores = (scores - scores.min()) / (scores.max() - scores.min())

    return scores, prior_turns


def select_informative_turns(
    dialogue: List[dict],
    turn_idx: int,
    kc: Optional[str],
    top_k: int = 5,
    threshold: float = 0.3,
):
    """
    Select the most informative turns based on IB-inspired scoring.

    Returns:
        - List of (turn, score, is_selected) tuples
        - n_selected: number of turns above threshold
    """
    scores, prior_turns = _compute_turn_informativeness(dialogue, turn_idx, kc)

    if len(scores) == 0:
        return [], 0

    # Select turns above threshold, limited to top_k
    above_threshold = scores >= threshold
    selected_indices = set()

    # Always include the most recent turn (strongest temporal signal)
    if len(scores) > 0:
        selected_indices.add(len(scores) - 1)

    # Add top-k by score
    sorted_indices = np.argsort(scores)[::-1]
    for idx in sorted_indices:
        if len(selected_indices) >= top_k:
            break
        if scores[idx] >= threshold:
            selected_indices.add(int(idx))

    results = []
    for i, t in enumerate(prior_turns):
        results.append((t, float(scores[i]), i in selected_indices))

    return results, len(selected_indices)


def format_ib_turn_dialogue(
    dialogue: List[dict],
    turn_idx: int,
    kc: Optional[str],
    sample: dict,
    args,
):
    """Build dialogue text with IB-selected informative turns marked."""
    turn_info, n_selected = select_informative_turns(
        dialogue, turn_idx, kc,
        top_k=getattr(args, "state_top_k", 5)
    )

    prompt = ""
    if args.dataset == "mathdial":
        prompt += get_mathdial_context(sample) + "\n\n"

    # Build turn map from prior turns to dialogue indices
    prior_turns_set = {t["turn"]: (t, is_sel) for t, _, is_sel in turn_info}

    # Show IB selection summary
    prompt += "[BEGIN EVIDENCE SELECTION]\n"
    prompt += f"Selected {n_selected} most informative prior turns "
    prompt += f"(out of {len(turn_info)} total) for predicting '{kc or 'this KC'}':\n"
    for t, score, is_sel in turn_info:
        marker = "★" if is_sel else " "
        prompt += f"  {marker} Turn {t['turn']}: score={score:.3f}, "
        prompt += f"correct={t.get('correct')}, "
        prompt += f"KCs={standards_to_str(t.get('kcs', []), ' ')}\n"
    prompt += "[END EVIDENCE SELECTION]\n\n"

    # Build dialogue text with markers on selected turns
    lines = []
    for turn in dialogue:
        is_sel = prior_turns_set.get(turn["turn"], (None, False))[1]
        marker_start = "<informative_turn>" if is_sel else ""
        marker_end = "</informative_turn>" if is_sel else ""

        if turn["teacher"]:
            lines.append(f"{marker_start}Teacher Turn {turn['turn']}: {turn['teacher']}{marker_end}")

        if turn_idx is not None and turn_idx == turn["turn"]:
            break

        if turn["student"]:
            lines.append(f"{marker_start}Student Turn {turn['turn']}: {turn['student']}{marker_end}")

    prompt += "[BEGIN DIALOGUE]\n" + "\n".join(lines) + "\n[END DIALOGUE]"
    return prompt


def kt_user_prompt_ib_turns(
    sample: dict,
    dialogue_anno: List[dict],
    turn_idx: int,
    kc: Optional[str],
    args,
):
    """IB-inspired user prompt with informative turn selection."""
    return format_ib_turn_dialogue(dialogue_anno, turn_idx, kc, sample, args)
