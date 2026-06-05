from typing import List, Optional
import numpy as np
from transformers import AutoTokenizer

from dialogue_kt.data_loading import correct_to_str, standards_to_str
from dialogue_kt.selfelicit import (
    MARKER_IMPSTART, MARKER_IMPEND,
    mark_evidence_in_dialogue_text
)

# ===== General functions =====

def get_dialogue_text(dialogue: List[dict], turn_idx: int = None, include_labels: bool = False, tag_wrapper: bool = True):
    lines = []
    for turn in dialogue:
        if turn["teacher"]:
            lines.append(f"Teacher Turn {turn['turn']}: {turn['teacher']}")
        # When turn_idx is given, stop after teacher utterance for that turn
        # Student utterance not included since would leak the correctness label in KT objective
        if turn_idx is not None and turn_idx == turn["turn"]:
            break
        if turn["student"]:
            lines.append(f"Student Turn {turn['turn']}: {turn['student']}")
        if include_labels:
            lines.append(f"Student Turn {turn['turn']} Correct: {correct_to_str(turn['correct'])}")
            lines.append(f"Turn {turn['turn']} Knowledge Components: {standards_to_str(turn['kcs'], ' ')}")
    prompt = "\n".join(lines)
    if tag_wrapper:
        prompt = "[BEGIN DIALOGUE]\n" + prompt + "\n[END DIALOGUE]"
    return prompt

def get_mathdial_context(sample: dict):
    return (f"[BEGIN PROBLEM]\n{sample['meta_data']['question'].strip()}\n[END PROBLEM]\n\n"
            f"[BEGIN CORRECT SOLUTION]\n{sample['meta_data']['correct_solution'].strip()}\n[END CORRECT SOLUTION]\n\n"
            f"[BEGIN INCORRECT STUDENT SOLUTION]\n{sample['meta_data']['incorrect_solution'].strip()}\n[END INCORRECT STUDENT SOLUTION]")

def get_true_false_tokens(tokenizer: AutoTokenizer):
    true = tokenizer("True").input_ids[-1]
    false = tokenizer("False").input_ids[-1]
    return true, false


# ===== Annotation prompting =====

COMTA_DIALOGUE_DESC = "the student is learning about math concepts."
MATHDIAL_DIALOGUE_DESC = "the student is attempting to solve a math problem. You are also given this problem, its correct solution, and the incorrect solution the student initially gave."

ANNO_BASE_SYSTEM_PROMPT = """You are an experienced math teacher and education expert. You are given a dialogue between a student and teacher where {desc} Your job is to list the math concepts/skills that can be used to classify the learning objectives at each turn in this dialogue. Please follow these instructions carefully when making your prediction:
- Each math concept/skill should be short description of a single learning objective. They should be generic enough so that they can be applied across dialogues and educational settings.
- When applicable, math/concepts skills should be repeated across turns.
- Teacher turns are often phrased as questions. In these cases, choose math concepts/skills that the student will need in order to respond correctly to the teacher's question.
- Do not give any math concepts/skills for turn 0 since there is no question being asked by the teacher.
- Your final response should have one entry for each teacher/student pair after turn 0.
- Before giving your final response, write a short summary of each turn in the dialogue, including the intended learning objectives.
- Along with each summary, list ALL candidate math concepts/skills that can be used to describe each turn in the dialogue.
- Your final response should be a JSON object using the template: result = {{"turn 1": ["math concept/skill 1", "math concept/skill 2", ...], "turn 2": ...}}"""

ANNO_CORRECTNESS_SYSTEM_PROMPT = """You are an experienced math teacher and education expert. You are given a dialogue between a student and teacher where {desc} Your job is to determine if the student has correctly answered each of the teacher's questions in the dialogue. Please follow these instructions carefully when making your prediction:
- The prediction should be "na" if the teacher asks a question that 1) does not attempt to assess the student's knowledge, or 2) does not necessarily have a right or wrong answer.
- If the question 1) does assess student knowledge and has a correct answer, and 2) the student does not respond directly to the question, then the prediction should be "false".
- Do not give a prediction for turn 0 since the teacher does not ask a question.
- Before giving your final response, write a short summary of each turn, including the intended learning objectives, and explain why the student's response is right or wrong.
- Your final prediction should be a JSON object using the template: result = {{"turn 1": "true/false/na", "turn 2": ...}}"""

ANNO_ATC_DOMAIN_SYSTEM_PROMPT = """You are an experienced math teacher and education expert. You are given a dialogue between a student and teacher where {desc} Your job is to list the common core domains that can be used to classify the learning objectives in this dialogue. Please follow these instructions carefully when making your prediction:
- You will be given a list of common core domains to choose from. When choosing them, write their names exactly as they appear. Do not use any domains that are not in this list.
- Before giving your final response, write a short summary of the dialogue.
- Your final response should be a list using the template: result = ["domain 1 name", "domain 2 name", ...]"""

ANNO_ATC_CLUSTER_SYSTEM_PROMPT = """You are an experienced math teacher and education expert. You are given a dialogue between a student and teacher where {desc} Your job is to list the common core math concepts/skills that can be used to classify the learning objectives in this dialogue. Please follow these instructions carefully when making your prediction:
- You will be given a list of common core math concepts/skills to choose from. When choosing them, write their IDs exactly as they appear. Do not use any math concepts/skills that are not in this list.
- Teacher turns are often phrased as questions. In these cases, choose math concepts/skills that the student will need in order to respond correctly to the teacher's question.
- Before giving your final response, write a short summary of each turn in the dialogue, including the intended learning objectives.
- Along with each summary, list ALL candidate math concepts/skills that can be used to describe each turn in the dialogue. If there are multiple math concepts/skills with the same description but different IDs and they both apply, then list both IDs.
- Your final response should be a list using the template: result = ["math concept/skill 1 id", "math concept/skill 2 id", ...]"""

ANNO_ATC_STANDARD_SYSTEM_PROMPT = """You are an experienced math teacher and education expert. You are given a dialogue between a student and teacher where {desc} Your job is to list the common core standards that can be used to classify the learning objectives at each turn in this dialogue. Please follow these instructions carefully when making your prediction:
- You will be given a list of common core standards to choose from. When choosing them, write their IDs exactly as they appear. Do not use any standards that are not in this list.
- Teacher turns are often phrased as questions. In these cases, choose standards that the student will need in order to respond correctly to the teacher's question.
- Do not give any standards for turn 0 since there is no question being asked by the teacher.
- Before giving your final response, write a short summary of each turn in the dialogue, including the intended learning objectives.
- Along with each summary, list ALL candidate standards that can be used to describe each turn in the dialogue. If there are multiple standards with the same description but different IDs and they both apply, then list both IDs.
- Your final response should be a JSON object using the template: result = {{"turn 1": ["standard 1 id", "standard 2 id", ...], "turn 2": ...}}"""

def get_dataset_desc(args):
    if args.dataset == "comta":
        return COMTA_DIALOGUE_DESC
    if args.dataset == "mathdial":
        return MATHDIAL_DIALOGUE_DESC
    raise Exception(f"No dataset description defined for {args.dataset}")

def anno_base_system_prompt(args):
    return ANNO_BASE_SYSTEM_PROMPT.format(desc=get_dataset_desc(args))

def anno_base_user_prompt(sample: dict, args):
    prompt = ""
    if args.dataset == "mathdial":
        prompt += get_mathdial_context(sample) + "\n\n"
    prompt += get_dialogue_text(sample["dialogue"])
    max_turn = sample["dialogue"][-1]["turn"]
    prompt += f"\n\nYour final response should have an entry for exactly {max_turn} turn{f's (1-{max_turn})' if max_turn > 1 else ''}."
    return prompt

def anno_atc_system_prompt(level: str, args):
    assert level in ("domain", "cluster", "standard")
    if level == "domain":
        return ANNO_ATC_DOMAIN_SYSTEM_PROMPT.format(desc=get_dataset_desc(args))
    if level == "cluster":
        return ANNO_ATC_CLUSTER_SYSTEM_PROMPT.format(desc=get_dataset_desc(args))
    return ANNO_ATC_STANDARD_SYSTEM_PROMPT.format(desc=get_dataset_desc(args))

def anno_atc_user_prompt(sample: dict, level: str, options: List[str], args):
    assert level in ("domain", "cluster", "standard")
    prompt = ""
    if args.dataset == "mathdial":
        prompt += get_mathdial_context(sample) + "\n\n"
    prompt += get_dialogue_text(sample["dialogue"])
    desc = "DOMAINS" if level == "domain" else "MATH CONCEPTS/SKILLS" if level == "cluster" else "STANDARDS"
    prompt += f"\n\n[BEGIN {desc}]\n- " + "\n- ".join(options) + f"\n[END {desc}]"
    if level == "standard":
        max_turn = sample["dialogue"][-1]["turn"]
        prompt += f"\n\nThere should be exactly {max_turn} turn{'s' if max_turn > 1 else ''} in your final result."
    return prompt

def anno_correctness_system_prompt(args):
    return ANNO_CORRECTNESS_SYSTEM_PROMPT.format(desc=get_dataset_desc(args))


# ===== KT model prompting =====

KT_SYSTEM_PROMPT = """You are an experienced math teacher. You are given a dialogue between a student and teacher where {desc} Your job is to predict if the student has a particular knowledge component at the current point in the dialogue. Please follow these instructions carefully when making your prediction:
- The student will need to possess this knowledge component in order to respond correctly to the teacher's most recent question.
- Use previous information in the dialogue to determine if the student has this knowledge component or not.
- Only respond with a single word, "True" or "False"."""

def kt_system_prompt(args):
    prompt = KT_SYSTEM_PROMPT.format(desc=get_dataset_desc(args))
    extra_instructions = []
    if getattr(args, "bayes_evidence", None):
        from dialogue_kt.bayesian_epistemology import bayesian_system_instruction
        extra_instructions.append(bayesian_system_instruction(
            include_labels=getattr(args, "bayes_evidence", None) == "adaptive_labels"
        ))
    if getattr(args, "informativeness", None):
        from dialogue_kt.informativeness_search import informativeness_system_instruction
        extra_instructions.append(informativeness_system_instruction())
    if extra_instructions:
        prompt = prompt[:-1] + "\n- " + "\n- ".join(extra_instructions) + "\n- Only respond with a single word, \"True\" or \"False\"."
    return prompt

def kt_user_prompt(sample: dict, dialogue_anno: List[dict], turn_idx: int, kc: Optional[str], args):
    prompt = ""
    if args.dataset == "mathdial":
        prompt += get_mathdial_context(sample) + "\n\n"
    prompt += get_dialogue_text(dialogue_anno, turn_idx=turn_idx, include_labels=args.prompt_inc_labels)
    prompt += f"\n\nKnowledge Component:"
    if kc:
        prompt += " " + kc
    return prompt


def get_dialogue_text_bayes(
    dialogue: List[dict],
    turn_idx: int = None,
    strengths: np.ndarray = None,
    include_labels: bool = False,
    tag_wrapper: bool = True,
):
    """Build dialogue text with Bayesian evidence reliability markers."""
    if strengths is None:
        return get_dialogue_text(dialogue, turn_idx, include_labels, tag_wrapper)

    from dialogue_kt.bayesian_epistemology import wrap_evidence

    lines = []
    for i, turn in enumerate(dialogue):
        strength = int(strengths[i]) if i < len(strengths) else 0
        if turn["teacher"]:
            lines.append(wrap_evidence(f"Teacher Turn {turn['turn']}: {turn['teacher']}", strength))
        if turn_idx is not None and turn_idx == turn["turn"]:
            break
        if turn["student"]:
            lines.append(wrap_evidence(f"Student Turn {turn['turn']}: {turn['student']}", strength))
        if include_labels:
            label_strength = max(strength, 1)
            lines.append(wrap_evidence(f"Student Turn {turn['turn']} Correct: {correct_to_str(turn['correct'])}", label_strength))
            lines.append(wrap_evidence(f"Turn {turn['turn']} Knowledge Components: {standards_to_str(turn['kcs'], ' ')}", label_strength))
    prompt = "\n".join(lines)
    if tag_wrapper:
        prompt = "[BEGIN DIALOGUE]\n" + prompt + "\n[END DIALOGUE]"
    return prompt


def kt_user_prompt_bayes(sample: dict, dialogue_anno: List[dict], turn_idx: int, kc: Optional[str], strengths: np.ndarray, args):
    """KT user prompt with Bayesian evidence reliability markers."""
    prompt = ""
    if args.dataset == "mathdial":
        prompt += get_mathdial_context(sample) + "\n\n"
    prompt += get_dialogue_text_bayes(
        dialogue_anno,
        turn_idx=turn_idx,
        strengths=strengths,
        include_labels=getattr(args, "bayes_evidence", None) == "adaptive_labels" or args.prompt_inc_labels,
    )
    prompt += f"\n\nKnowledge Component:"
    if kc:
        prompt += " " + kc
    return prompt


def get_dialogue_text_informativeness(
    dialogue: List[dict],
    turn_idx: int = None,
    strengths: np.ndarray = None,
    include_labels: bool = False,
    tag_wrapper: bool = True,
    prune: bool = False,
):
    """Build dialogue text with stepwise informativeness markers or pruning."""
    if strengths is None:
        return get_dialogue_text(dialogue, turn_idx, include_labels, tag_wrapper)

    from dialogue_kt.informativeness_search import wrap_step

    lines = []
    for i, turn in enumerate(dialogue):
        strength = int(strengths[i]) if i < len(strengths) else 0
        is_current = turn_idx is not None and turn_idx == turn["turn"]
        if prune and strength == 0 and not is_current:
            continue
        if turn["teacher"]:
            line = f"Teacher Turn {turn['turn']}: {turn['teacher']}"
            lines.append(line if prune else wrap_step(line, strength))
        if is_current:
            break
        if turn["student"]:
            line = f"Student Turn {turn['turn']}: {turn['student']}"
            lines.append(line if prune else wrap_step(line, strength))
        if include_labels:
            lines.append(f"Student Turn {turn['turn']} Correct: {correct_to_str(turn['correct'])}")
            lines.append(f"Turn {turn['turn']} Knowledge Components: {standards_to_str(turn['kcs'], ' ')}")
    prompt = "\n".join(lines)
    if tag_wrapper:
        prompt = "[BEGIN DIALOGUE]\n" + prompt + "\n[END DIALOGUE]"
    return prompt


def kt_user_prompt_informativeness(sample: dict, dialogue_anno: List[dict], turn_idx: int, kc: Optional[str], strengths: np.ndarray, args):
    """KT user prompt with stepwise informativeness markers."""
    prompt = ""
    if args.dataset == "mathdial":
        prompt += get_mathdial_context(sample) + "\n\n"
    info_mode = getattr(args, "informativeness", None) or ""
    prompt += get_dialogue_text_informativeness(
        dialogue_anno,
        turn_idx=turn_idx,
        strengths=strengths,
        include_labels=args.prompt_inc_labels,
        prune=info_mode.endswith("_prune"),
    )
    prompt += f"\n\nKnowledge Component:"
    if kc:
        prompt += " " + kc
    return prompt

def dkt_sem_prompt(teacher_turn: str, student_turn: str, kcs: List[str], correct: bool):
    return f"""A teacher and a student are having a dialogue about math concepts. Below is a single turn pair from that dialouge, where the student responds to the teacher. In addition, there are knowledge components that represent the learning objectives in the teacher's question. Finally, it is identified if the student's response to the teacher was correct or incorrect.
Teacher: {teacher_turn}
Student: {student_turn}
Knowledge Components: {standards_to_str(kcs, ' ')}
Student Correct: {correct_to_str(correct)}"""


# ===== SELFELICIT Prompt Templates =====

KT_SYSTEM_PROMPT_SE = """You are an experienced math teacher. You are given a dialogue between a student and teacher where {desc} Your job is to predict if the student has a particular knowledge component at the current point in the dialogue. Please follow these instructions carefully when making your prediction:
- The student will need to possess this knowledge component in order to respond correctly to the teacher's most recent question.
- Use previous information in the dialogue to determine if the student has this knowledge component or not.
- Within the dialogue, {marker_impstart} and {marker_impend} markers indicate the most important evidence turns. Pay special attention to these marked turns when making your prediction.
- Only respond with a single word, "True" or "False"."""

def kt_system_prompt_se(args):
    """SELFELICIT-enhanced system prompt that instructs model to use evidence markers."""
    desc = get_dataset_desc(args)
    return KT_SYSTEM_PROMPT_SE.format(
        desc=desc,
        marker_impstart=MARKER_IMPSTART,
        marker_impend=MARKER_IMPEND
    )

def get_dialogue_text_se(
    dialogue: List[dict],
    turn_idx: int = None,
    evidence_mask: np.ndarray = None,
    include_labels: bool = False,
    tag_wrapper: bool = True,
    marker_start: str = MARKER_IMPSTART,
    marker_end: str = MARKER_IMPEND
):
    """
    Build dialogue text with SELFELICIT evidence markers around high-importance turns.

    Args:
        dialogue: List of turn dicts with teacher/student text
        turn_idx: If set, stop after this turn (for KT prediction at this point)
        evidence_mask: Boolean array marking which turns are evidence
        include_labels: Whether to include correctness/KC labels
        tag_wrapper: Whether to wrap in [BEGIN/END DIALOGUE] tags
        marker_start: Start marker for evidence
        marker_end: End marker for evidence

    Returns:
        Marked dialogue string
    """
    if evidence_mask is None:
        # Fall back to standard dialogue text
        return get_dialogue_text(dialogue, turn_idx, include_labels, tag_wrapper)

    # Build turn index set for evidence turns
    evidence_turn_indices = set(
        i for i, is_ev in enumerate(evidence_mask) if is_ev
    )

    lines = []
    for i, turn in enumerate(dialogue):
        is_evidence = i in evidence_turn_indices

        if turn["teacher"]:
            teacher_line = f"Teacher Turn {turn['turn']}: {turn['teacher']}"
            lines.append(
                f"{marker_start} {teacher_line} {marker_end}" if is_evidence else teacher_line
            )

        # When turn_idx is given, stop after teacher utterance for that turn
        if turn_idx is not None and turn_idx == turn["turn"]:
            break

        if turn["student"]:
            student_line = f"Student Turn {turn['turn']}: {turn['student']}"
            lines.append(
                f"{marker_start} {student_line} {marker_end}" if is_evidence else student_line
            )

        if include_labels:
            lines.append(f"Student Turn {turn['turn']} Correct: {correct_to_str(turn['correct'])}")
            lines.append(f"Turn {turn['turn']} Knowledge Components: {standards_to_str(turn['kcs'], ' ')}")

    prompt = "\n".join(lines)
    if tag_wrapper:
        prompt = "[BEGIN DIALOGUE]\n" + prompt + "\n[END DIALOGUE]"
    return prompt


def kt_user_prompt_se(
    sample: dict,
    dialogue_anno: List[dict],
    turn_idx: int,
    kc: Optional[str],
    evidence_mask: np.ndarray,
    args
):
    """
    SELFELICIT-enhanced user prompt with evidence-marked dialogue turns.

    Args:
        sample: Raw sample dict with meta_data
        dialogue_anno: Annotated dialogue turns
        turn_idx: Current turn for prediction
        kc: Knowledge component text (None for base prompt)
        evidence_mask: Boolean array of evidence turns
        args: Training args

    Returns:
        SE-enhanced prompt string
    """
    prompt = ""
    if args.dataset == "mathdial":
        prompt += get_mathdial_context(sample) + "\n\n"
    prompt += get_dialogue_text_se(
        dialogue_anno,
        turn_idx=turn_idx,
        evidence_mask=evidence_mask,
        include_labels=args.prompt_inc_labels
    )
    prompt += f"\n\nKnowledge Component:"
    if kc:
        prompt += " " + kc
    return prompt
