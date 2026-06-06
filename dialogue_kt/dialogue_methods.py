from collections import Counter, defaultdict
import math
import re
from typing import List, Optional, Tuple

from dialogue_kt.prompting import get_dialogue_text, get_mathdial_context, correct_to_str, standards_to_str


def method_system_instruction(args):
    method = getattr(args, "kt_method", "base")
    if method == "state_table":
        return "Use the past KC state table as causal knowledge tracing evidence. It contains only prior turns, never the current student answer. Prefer repeated recent incorrect evidence over superficial lexical similarity."
    if method == "solution_contrast":
        return "Compare the visible dialogue with the correct solution and the initial incorrect student solution. Predict mastery only when the visible dialogue suggests the student has moved toward the correct reasoning for the requested knowledge component."
    if method == "support_token":
        return "Use the support summary to decide whether prior dialogue evidence is sufficient for the requested knowledge component. Be conservative when evidence is unsupported or only weakly supported."
    if method == "hyper_chain":
        return "Use the valid evidence chain and ignore the distractor chain. Decide whether the student's visible reasoning chain supports mastery of the requested knowledge component."
    if method == "quito_mark":
        return "Use the <relevant_context> markers as query-conditioned evidence for the current teacher question and knowledge components. Unmarked context may still be useful background."
    if method == "dac_mark":
        return "Use the <important_context> markers as compressed high-value dialogue evidence while preserving the full causal dialogue context."
    if method == "dual_view_consistency":
        return "Make the same knowledge tracing judgment from raw dialogue and structured evidence views. Treat all evidence as causal history only."
    return ""


def _visible_turns(dialogue: List[dict], turn_idx: int):
    return [turn for turn in dialogue if turn["turn"] < turn_idx]


def _tokenize(text: str):
    return set(re.findall(r"[a-zA-Z0-9]+", (text or "").lower()))


def _overlap(a: str, b: str):
    ta, tb = _tokenize(a), _tokenize(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / max(1, len(ta | tb))


def build_past_kc_state(dialogue: List[dict], turn_idx: int, top_k: int = 8):
    rows = {}
    for turn in _visible_turns(dialogue, turn_idx):
        if turn.get("correct") is None:
            continue
        for kc in turn.get("kcs", []):
            row = rows.setdefault(kc, {
                "kc": kc,
                "prior_attempts": 0,
                "prior_correct": 0,
                "prior_incorrect": 0,
                "last_result": "none",
                "turns_since_last": 999,
                "last_teacher_question": "",
            })
            row["prior_attempts"] += 1
            if turn["correct"]:
                row["prior_correct"] += 1
                row["last_result"] = "correct"
            else:
                row["prior_incorrect"] += 1
                row["last_result"] = "incorrect"
            row["turns_since_last"] = turn_idx - turn["turn"]
            row["last_teacher_question"] = (turn.get("teacher") or "")[:180]
    ranked = sorted(rows.values(), key=lambda r: (r["turns_since_last"], -r["prior_attempts"], -r["prior_incorrect"]))
    return ranked[:top_k]


def format_state_table(rows):
    if not rows:
        return "[BEGIN PAST KC STATE]\nNo prior KC attempts are visible before the current teacher question.\n[END PAST KC STATE]"
    lines = ["[BEGIN PAST KC STATE]"]
    for i, row in enumerate(rows, 1):
        lines.append(
            f"{i}. KC: {row['kc']}\n"
            f"   prior_attempts: {row['prior_attempts']}; prior_correct: {row['prior_correct']}; prior_incorrect: {row['prior_incorrect']}; "
            f"last_result: {row['last_result']}; turns_since_last: {row['turns_since_last']}\n"
            f"   last_teacher_question: {row['last_teacher_question']}"
        )
    lines.append("[END PAST KC STATE]")
    return "\n".join(lines)


def kt_user_prompt_state_table(sample: dict, dialogue_anno: List[dict], turn_idx: int, kc: Optional[str], args):
    prompt = ""
    if args.dataset == "mathdial":
        prompt += get_mathdial_context(sample) + "\n\n"
    prompt += format_state_table(build_past_kc_state(dialogue_anno, turn_idx, getattr(args, "state_top_k", 8)))
    prompt += "\n\n" + get_dialogue_text(dialogue_anno, turn_idx=turn_idx, include_labels=False)
    prompt += "\n\nKnowledge Component:"
    if kc:
        prompt += " " + kc
    return prompt


def solution_contrast_signals(sample: dict, dialogue: List[dict], turn_idx: int):
    meta = sample.get("meta_data", {})
    correct_solution = meta.get("correct_solution", "")
    incorrect_solution = meta.get("incorrect_solution", "")
    visible_text = " ".join(
        f"{t.get('teacher','')} {t.get('student','')}" for t in dialogue if t["turn"] < turn_idx
    )
    current_teacher = " ".join(t.get("teacher", "") for t in dialogue if t["turn"] == turn_idx)
    corr_ov = _overlap(visible_text + " " + current_teacher, correct_solution)
    inc_ov = _overlap(visible_text + " " + current_teacher, incorrect_solution)
    def band(x):
        return "high" if x >= 0.18 else "medium" if x >= 0.08 else "low"
    nums_vis = set(re.findall(r"-?\d+(?:\.\d+)?", visible_text + " " + current_teacher))
    nums_corr = set(re.findall(r"-?\d+(?:\.\d+)?", correct_solution))
    nums_inc = set(re.findall(r"-?\d+(?:\.\d+)?", incorrect_solution))
    return {
        "correct_overlap": band(corr_ov),
        "incorrect_overlap": band(inc_ov),
        "shared_correct_numbers": len(nums_vis & nums_corr),
        "shared_incorrect_numbers": len(nums_vis & nums_inc),
    }


def format_solution_contrast(sample: dict, dialogue: List[dict], turn_idx: int):
    sig = solution_contrast_signals(sample, dialogue, turn_idx)
    return (
        "[BEGIN SOLUTION CONTRAST SIGNALS]\n"
        f"Visible dialogue overlap with correct solution: {sig['correct_overlap']}\n"
        f"Visible dialogue overlap with initial incorrect solution: {sig['incorrect_overlap']}\n"
        f"Shared numbers with correct solution: {sig['shared_correct_numbers']}\n"
        f"Shared numbers with initial incorrect solution: {sig['shared_incorrect_numbers']}\n"
        "[END SOLUTION CONTRAST SIGNALS]"
    )


def kt_user_prompt_solution_contrast(sample: dict, dialogue_anno: List[dict], turn_idx: int, kc: Optional[str], args):
    prompt = ""
    if args.dataset == "mathdial":
        prompt += get_mathdial_context(sample) + "\n\n"
    prompt += format_solution_contrast(sample, dialogue_anno, turn_idx)
    prompt += "\n\n" + get_dialogue_text(dialogue_anno, turn_idx=turn_idx, include_labels=False)
    prompt += "\n\nKnowledge Component:"
    if kc:
        prompt += " " + kc
    return prompt


def support_summary(dialogue: List[dict], turn_idx: int):
    prior = _visible_turns(dialogue, turn_idx)
    attempts = [t for t in prior if t.get("correct") is not None]
    correct = sum(1 for t in attempts if t.get("correct"))
    incorrect = sum(1 for t in attempts if t.get("correct") is False)
    recent = attempts[-3:]
    recent_correct = sum(1 for t in recent if t.get("correct"))
    support = "strong" if len(attempts) >= 3 else "weak" if attempts else "unsupported"
    return (
        "[BEGIN SUPPORT SUMMARY]\n"
        f"Evidence support level before current question: {support}\n"
        f"Prior labeled attempts: {len(attempts)}; prior correct: {correct}; prior incorrect: {incorrect}; recent correct among last 3: {recent_correct}\n"
        "[END SUPPORT SUMMARY]"
    )


def kt_user_prompt_support_token(sample: dict, dialogue_anno: List[dict], turn_idx: int, kc: Optional[str], args):
    prompt = ""
    if args.dataset == "mathdial":
        prompt += get_mathdial_context(sample) + "\n\n"
    prompt += support_summary(dialogue_anno, turn_idx)
    prompt += "\n\n" + get_dialogue_text(dialogue_anno, turn_idx=turn_idx, include_labels=False)
    prompt += "\n\nKnowledge Component:"
    if kc:
        prompt += " " + kc
    return prompt


def evidence_chain_blocks(dialogue: List[dict], turn_idx: int):
    prior = [t for t in _visible_turns(dialogue, turn_idx) if t.get("correct") is not None]
    valid = prior[-3:]
    distractor = list(reversed(prior[:3]))
    def fmt(name, turns):
        lines = [f"[BEGIN {name}]"]
        if not turns:
            lines.append("No prior labeled turns.")
        for t in turns:
            lines.append(f"Turn {t['turn']}: correct={correct_to_str(t.get('correct'))}; KCs={standards_to_str(t.get('kcs', []), ' ')}; Student={t.get('student','')[:160]}")
        lines.append(f"[END {name}]")
        return "\n".join(lines)
    return fmt("VALID EVIDENCE CHAIN", valid) + "\n\n" + fmt("DISTRACTOR CHAIN", distractor)


def kt_user_prompt_hyper_chain(sample: dict, dialogue_anno: List[dict], turn_idx: int, kc: Optional[str], args):
    prompt = ""
    if args.dataset == "mathdial":
        prompt += get_mathdial_context(sample) + "\n\n"
    prompt += evidence_chain_blocks(dialogue_anno, turn_idx)
    prompt += "\n\n" + get_dialogue_text(dialogue_anno, turn_idx=turn_idx, include_labels=False)
    prompt += "\n\nKnowledge Component:"
    if kc:
        prompt += " " + kc
    return prompt


def relevance_scores(dialogue: List[dict], turn_idx: int):
    current = next((t for t in dialogue if t["turn"] == turn_idx), None)
    query = (current or {}).get("teacher", "") + " " + " ".join((current or {}).get("kcs", []))
    scores = []
    for i, t in enumerate(dialogue):
        if t["turn"] > turn_idx:
            scores.append(0.0)
            continue
        txt = f"{t.get('teacher','')} {t.get('student','')} {' '.join(t.get('kcs', []))}"
        recency = 1.0 / (1.0 + max(0, turn_idx - t["turn"]))
        scores.append(_overlap(query, txt) + 0.08 * recency)
    return scores


def marked_dialogue_text(dialogue: List[dict], turn_idx: int, marker_start: str, marker_end: str, mode: str):
    scores = relevance_scores(dialogue, turn_idx)
    positive = [(i, s) for i, s in enumerate(scores) if s > 0]
    top = {i for i, _ in sorted(positive, key=lambda x: x[1], reverse=True)[:3]}
    lines = []
    for i, turn in enumerate(dialogue):
        mark = i in top or turn["turn"] == turn_idx
        if turn.get("teacher"):
            line = f"Teacher Turn {turn['turn']}: {turn['teacher']}"
            lines.append(f"{marker_start} {line} {marker_end}" if mark else line)
        if turn_idx is not None and turn_idx == turn["turn"]:
            break
        if turn.get("student"):
            line = f"Student Turn {turn['turn']}: {turn['student']}"
            lines.append(f"{marker_start} {line} {marker_end}" if mark else line)
    return "[BEGIN DIALOGUE]\n" + "\n".join(lines) + "\n[END DIALOGUE]"


def kt_user_prompt_marked(sample: dict, dialogue_anno: List[dict], turn_idx: int, kc: Optional[str], args, marker_start: str, marker_end: str):
    prompt = ""
    if args.dataset == "mathdial":
        prompt += get_mathdial_context(sample) + "\n\n"
    prompt += marked_dialogue_text(dialogue_anno, turn_idx, marker_start, marker_end, getattr(args, "kt_method", ""))
    prompt += "\n\nKnowledge Component:"
    if kc:
        prompt += " " + kc
    return prompt


def kt_user_prompt_method(sample: dict, dialogue_anno: List[dict], turn_idx: int, kc: Optional[str], args):
    method = getattr(args, "kt_method", "base")
    if method == "state_table" or (method == "dual_view_consistency" and getattr(args, "dual_view_type", "state_table") == "state_table"):
        return kt_user_prompt_state_table(sample, dialogue_anno, turn_idx, kc, args)
    if method == "solution_contrast" or (method == "dual_view_consistency" and getattr(args, "dual_view_type", "state_table") == "solution_contrast"):
        return kt_user_prompt_solution_contrast(sample, dialogue_anno, turn_idx, kc, args)
    if method == "support_token":
        return kt_user_prompt_support_token(sample, dialogue_anno, turn_idx, kc, args)
    if method == "hyper_chain":
        return kt_user_prompt_hyper_chain(sample, dialogue_anno, turn_idx, kc, args)
    if method == "quito_mark":
        return kt_user_prompt_marked(sample, dialogue_anno, turn_idx, kc, args, "<relevant_context>", "</relevant_context>")
    if method == "dac_mark":
        return kt_user_prompt_marked(sample, dialogue_anno, turn_idx, kc, args, "<important_context>", "</important_context>")
    raise ValueError(f"Unsupported kt_method prompt: {method}")
