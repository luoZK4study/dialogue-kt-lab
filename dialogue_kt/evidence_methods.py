"""
Evidence-R1 and HypER inspired methods for Dialogue-KT.

Evidence-R1 adaptation: Dual-process reasoning with explicit evidence
  reasoning + implicit self-reflection for KC mastery prediction.

HypER adaptation: Multi-hop evidence chain validation for dialogue
  turn relevance assessment.
"""
from typing import List, Optional
import numpy as np
import re
from collections import defaultdict

from dialogue_kt.prompting import get_dialogue_text, get_mathdial_context, correct_to_str, standards_to_str


# ============================================================
# Evidence-R1 Inspired: Dual-Process Reasoning
# ============================================================

def method_system_instruction_evidence_r1(args):
    return (
        "Before predicting, first identify which prior dialogue turns provide "
        "evidence about the student's mastery of the requested knowledge component. "
        "Then evaluate how strongly this evidence supports or contradicts mastery. "
        "Finally, predict True or False based on the preponderance of evidence."
    )


def format_evidence_reasoning_prompt(
    dialogue: List[dict],
    turn_idx: int,
    kc: Optional[str],
    sample: dict,
    args
):
    """
    Build an evidence-reasoning prompt that guides the model to:
    1. Identify relevant evidence turns
    2. Evaluate support strength
    3. Predict based on evidence preponderance
    """
    prior_turns = [t for t in dialogue if t["turn"] < turn_idx and t.get("correct") is not None]

    # Build evidence summary for each prior turn
    evidence_lines = ["[BEGIN EVIDENCE SUMMARY]"]
    for t in prior_turns:
        kc_match = any(kc.lower() in tk.lower() for tk in t.get("kcs", [])) if kc else False
        correctness = "correct" if t.get("correct") else "incorrect"
        relevance = "direct" if kc_match else "indirect"

        evidence_lines.append(
            f"Turn {t['turn']}: {correctness}, relevance={relevance}\n"
            f"  Teacher: {(t.get('teacher') or '')[:150]}\n"
            f"  Student: {(t.get('student') or '')[:150]}"
        )
    evidence_lines.append("[END EVIDENCE SUMMARY]")

    # Reasoning template
    reasoning_template = (
        "[BEGIN REASONING]\n"
        "Step 1 - Identify supporting evidence: Which turns suggest the student "
        "HAS mastered '{kc}'?\n"
        "Step 2 - Identify contradicting evidence: Which turns suggest the student "
        "has NOT mastered '{kc}'?\n"
        "Step 3 - Weigh the evidence: Is the supporting evidence stronger than "
        "contradicting evidence?\n"
        "Step 4 - Final judgment: Based on the evidence preponderance, predict "
        "True or False.\n"
        "[END REASONING]"
    ).format(kc=kc or "this knowledge component")

    prompt = ""
    if args.dataset == "mathdial":
        prompt += get_mathdial_context(sample) + "\n\n"
    prompt += "\n".join(evidence_lines)
    prompt += "\n\n" + reasoning_template
    prompt += "\n\n" + get_dialogue_text(dialogue, turn_idx=turn_idx, include_labels=False)
    return prompt


def kt_user_prompt_evidence_r1(
    sample: dict,
    dialogue_anno: List[dict],
    turn_idx: int,
    kc: Optional[str],
    args
):
    """Evidence-R1 style user prompt with explicit evidence reasoning."""
    return format_evidence_reasoning_prompt(dialogue_anno, turn_idx, kc, sample, args)


# ============================================================
# HypER Inspired: Multi-hop Evidence Chain Validation  
# ============================================================

def method_system_instruction_hyper_validate(args):
    return (
        "Evaluate the evidence chain formed by historical dialogue turns. "
        "A valid evidence chain shows a clear progression of the student's "
        "understanding. Invalid chains contain contradictory or irrelevant turns. "
        "Base your prediction only on the valid evidence chain."
    )


def build_evidence_chain(
    dialogue: List[dict],
    turn_idx: int,
    kc: Optional[str],
    top_k: int = 5
):
    """
    Build a multi-hop evidence chain from historical turns.

    Each hop represents a turn that provides evidence about the KC.
    The chain is ordered by relevance and temporal proximity.

    Returns:
        (valid_chain, invalid_turns, chain_score)
    """
    prior_turns = [t for t in dialogue if t["turn"] < turn_idx and t.get("correct") is not None]
    if not prior_turns:
        return [], [], 0.0

    # Score each turn by:
    # 1. KC relevance: Does the turn involve this KC or similar ones?
    # 2. Informativeness: Does the correctness label carry information?
    # 3. Recency: More recent turns are more relevant
    # 4. Consistency: Does this turn align with the trend?

    scored_turns = []
    for i, t in enumerate(prior_turns):
        # KC relevance
        turn_kcs = " ".join(t.get("kcs", [])).lower()
        kc_text = (kc or "").lower()
        kc_overlap = len(set(turn_kcs.split()) & set(kc_text.split())) / max(1, len(set(kc_text.split())))

        # Recency weight
        recency = 1.0 / (1.0 + turn_idx - t["turn"])

        # Informativeness: incorrect turns carry more information (surprise)
        informativeness = 0.7 if t.get("correct") is False else 0.3

        # Combined score
        score = 0.4 * kc_overlap + 0.3 * recency + 0.3 * informativeness
        scored_turns.append((t, score))

    # Sort by score descending
    scored_turns.sort(key=lambda x: x[1], reverse=True)

    # Top-k form the valid chain, rest are potentially distracting
    valid_chain = [t for t, s in scored_turns[:top_k]]
    invalid_turns = [t for t, s in scored_turns[top_k:]]

    # Chain coherence score: how consistently do the chain turns point in one direction?
    if len(valid_chain) >= 2:
        correctness_values = [1 if t.get("correct") else 0 for t in valid_chain]
        # Low variance = high coherence
        chain_score = 1.0 - np.var(correctness_values) if np.var(correctness_values) < 0.25 else 0.5
    else:
        chain_score = 0.5

    return valid_chain, invalid_turns, chain_score


def format_evidence_chain_prompt(
    dialogue: List[dict],
    turn_idx: int,
    kc: Optional[str],
    sample: dict,
    args
):
    """Build a prompt showing the validated evidence chain."""
    valid_chain, invalid_turns, chain_score = build_evidence_chain(
        dialogue, turn_idx, kc, getattr(args, "state_top_k", 5)
    )

    prompt = ""
    if args.dataset == "mathdial":
        prompt += get_mathdial_context(sample) + "\n\n"

    # Show the valid evidence chain
    prompt += "[BEGIN VALID EVIDENCE CHAIN]\n"
    prompt += f"Chain coherence score: {chain_score:.2f}\n"
    if valid_chain:
        for i, t in enumerate(valid_chain):
            correctness = "✓" if t.get("correct") else "✗"
            prompt += (
                f"Hop {i+1} — Turn {t['turn']} [{correctness}]: "
                f"{(t.get('teacher') or '')[:120]} → {(t.get('student') or '')[:120]}\n"
            )
    else:
        prompt += "No prior evidence available.\n"
    prompt += "[END VALID EVIDENCE CHAIN]\n\n"

    # Note about filtered turns
    if invalid_turns:
        prompt += (
            f"[NOTE: {len(invalid_turns)} earlier turns were filtered as "
            f"less relevant to '{kc or 'this KC'}'.]\n\n"
        )

    prompt += get_dialogue_text(dialogue, turn_idx=turn_idx, include_labels=False)
    return prompt


def kt_user_prompt_hyper_validate(
    sample: dict,
    dialogue_anno: List[dict],
    turn_idx: int,
    kc: Optional[str],
    args
):
    """HypER-style user prompt with evidence chain validation."""
    return format_evidence_chain_prompt(dialogue_anno, turn_idx, kc, sample, args)


# ============================================================
# Loss Functions
# ============================================================

def get_evidence_consistency_loss(model, batch, true_token, false_token, args):
    """
    Evidence-R1 inspired consistency loss.
    
    Trains the model to produce consistent predictions when 
    evidence is presented in different formats (raw dialogue vs. structured evidence).
    """
    from dialogue_kt.training import get_lmkt_loss_packed, aggregate_kc_probs
    from dialogue_kt.training import get_lmkt_probs_packed

    # Primary prediction (standard prompt)
    kc_probs1, kc_probs_grouped1 = get_lmkt_probs_packed(model, batch, true_token, false_token, args)
    corr_probs1 = aggregate_kc_probs(kc_probs1, batch, args)
    loss1 = torch.nn.BCELoss()(corr_probs1.float(), batch["labels"])

    # Secondary prediction (evidence-structured prompt, if available)
    if "input_ids_view2" in batch:
        kc_probs2, _ = get_lmkt_probs_packed(model, batch, true_token, false_token, args, suffix="_view2")
        corr_probs2 = aggregate_kc_probs(kc_probs2, batch, args, suffix="_view2")
        loss2 = torch.nn.BCELoss()(corr_probs2.float(), batch["labels"])

        # Consistency: predictions should agree regardless of evidence format
        consistency = torch.nn.functional.mse_loss(corr_probs1, corr_probs2)
        loss = 0.5 * (loss1 + loss2) + 0.1 * consistency
    else:
        loss = loss1

    return loss, kc_probs_grouped1, corr_probs1


# Import torch here for the loss function
import torch


# ============================================================
# Method Registry
# ============================================================

EVIDENCE_METHOD_PROMPTS = {
    "evidence_r1": kt_user_prompt_evidence_r1,
    "hyper_validate": kt_user_prompt_hyper_validate,
}

EVIDENCE_METHOD_INSTRUCTIONS = {
    "evidence_r1": method_system_instruction_evidence_r1,
    "hyper_validate": method_system_instruction_hyper_validate,
}
