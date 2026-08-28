"""
SELFELICIT: Your Language Model Secretly Knows Where is the Relevant Evidence.

Adapted for Dialogue Knowledge Tracing.
Core idea: Extract attention scores from deeper transformer layers to identify
which dialogue turns are most relevant evidence for KC prediction.

Reference: Liu et al. (ACL 2025) — https://arxiv.org/abs/2502.08767
"""

import torch
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional
from transformers import PreTrainedTokenizer

from dialogue_kt.utils import device

# Default markers matching SELFELICIT paper
MARKER_IMPSTART = "<start_important>"
MARKER_IMPEND = "<end_important>"
MARKER_SPANSTART = "<evidence>"
MARKER_SPANEND = "</evidence>"

# Default layer span: use deeper 50% of layers as evidence-reading layers
DEFAULT_LAYER_SPAN = (0.5, 1.0)

# Default threshold: sentences with score >= alpha * max_score are selected as evidence
DEFAULT_ALPHA = 0.5

SPAN_KEYWORDS = [
    "because", "so", "therefore", "equal", "equals", "equivalent",
    "fraction", "fractions", "denominator", "numerator", "multiply",
    "multiplied", "divide", "divided", "simplify", "simplified",
    "common denominator", "same number", "incorrect", "correct",
    "should be", "need to", "remember", "check", "reduce", "rewrite",
]


def merge_overlapping_spans(spans: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if not spans:
        return []
    spans = sorted((max(0, s), max(0, e)) for s, e in spans if e > s)
    merged = [spans[0]]
    for start, end in spans[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def extract_evidence_spans_from_utterance(text: str, max_spans: int = 4) -> List[Tuple[int, int]]:
    """Heuristic span extraction for phrases / short clauses / math expressions."""
    import re

    if not text:
        return []

    spans: List[Tuple[int, int]] = []

    # Short math expressions: fractions, equations, arithmetic snippets.
    patterns = [
        r"\b\d+\s*/\s*\d+\b",
        r"\b\d+(?:\.\d+)?\s*(?:=|==|!=|<=|>=|<|>|\+|\-|\*|x|×|/)\s*\d+(?:\.\d+)?(?:\s*(?:=|\+|\-|\*|x|×|/)\s*\d+(?:\.\d+)?)?",
        r"\b\d+(?:\.\d+)?\b",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            if match.end() - match.start() <= 32:
                spans.append((match.start(), match.end()))

    # Clause-level keywords.
    lowered = text.lower()
    for keyword in SPAN_KEYWORDS:
        start = 0
        while True:
            idx = lowered.find(keyword, start)
            if idx < 0:
                break
            clause_start = max(text.rfind(sep, 0, idx) for sep in [".", ",", ";", ":", "\n"])
            clause_start = 0 if clause_start < 0 else clause_start + 1
            clause_end_candidates = [text.find(sep, idx) for sep in [".", ",", ";", ":", "\n"] if text.find(sep, idx) >= 0]
            clause_end = min(clause_end_candidates) if clause_end_candidates else len(text)
            if 0 < clause_end - clause_start <= 96:
                spans.append((clause_start, clause_end))
            start = idx + len(keyword)

    return merge_overlapping_spans(spans)[:max_spans]


def apply_span_markers(text: str,
                       spans: List[Tuple[int, int]],
                       marker_start: str = MARKER_SPANSTART,
                       marker_end: str = MARKER_SPANEND) -> str:
    if not text or not spans:
        return text
    spans = merge_overlapping_spans(spans)
    pieces = []
    cursor = 0
    for start, end in spans:
        start = max(0, min(start, len(text)))
        end = max(start, min(end, len(text)))
        if start > cursor:
            pieces.append(text[cursor:start])
        pieces.append(f"{marker_start}{text[start:end]}{marker_end}")
        cursor = end
    if cursor < len(text):
        pieces.append(text[cursor:])
    return "".join(pieces)


def find_subsequence_positions(sequence: List[int], pattern: List[int]) -> List[Tuple[int, int]]:
    if not sequence or not pattern or len(pattern) > len(sequence):
        return []
    positions = []
    plen = len(pattern)
    for i in range(len(sequence) - plen + 1):
        if sequence[i:i + plen] == pattern:
            positions.append((i, i + plen))
    return positions


def build_evidence_token_mask_from_markers(input_ids: torch.Tensor, tokenizer: PreTrainedTokenizer) -> torch.Tensor:
    """Return boolean mask for tokens enclosed by turn/span evidence markers."""
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)

    marker_token_pairs = [
        (
            tokenizer.encode(f"{MARKER_IMPSTART} ", add_special_tokens=False),
            tokenizer.encode(f" {MARKER_IMPEND}", add_special_tokens=False),
        ),
        (
            tokenizer.encode(MARKER_SPANSTART, add_special_tokens=False),
            tokenizer.encode(MARKER_SPANEND, add_special_tokens=False),
        ),
    ]

    mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for b in range(input_ids.shape[0]):
        seq = input_ids[b].tolist()
        for start_tokens, end_tokens in marker_token_pairs:
            if not start_tokens or not end_tokens:
                continue
            start_positions = find_subsequence_positions(seq, start_tokens)
            end_positions = find_subsequence_positions(seq, end_tokens)
            end_cursor = 0
            for start_idx, start_end in start_positions:
                while end_cursor < len(end_positions) and end_positions[end_cursor][0] < start_end:
                    end_cursor += 1
                if end_cursor >= len(end_positions):
                    break
                end_start, _ = end_positions[end_cursor]
                if start_end < end_start:
                    mask[b, start_end:end_start] = True
                end_cursor += 1
    return mask


def get_evidence_layer_range(n_layers: int, layer_span: Tuple[float, float] = DEFAULT_LAYER_SPAN) -> Tuple[int, int]:
    """
    Compute the range of evidence-reading layers based on layer_span fraction.

    Args:
        n_layers: Total number of transformer layers in the model
        layer_span: (start_frac, end_frac) — fraction of layers to use

    Returns:
        (start_layer, end_layer) — integer layer indices
    """
    start = int(layer_span[0] * n_layers)
    end = int(layer_span[1] * n_layers)
    return start, end


def extract_attention_scores(
    outputs,
    layer_range: Tuple[int, int],
    context_span: Tuple[int, int],
    query_position: int = -1
) -> np.ndarray:
    """
    Extract and aggregate attention scores from evidence-reading layers.

    Following the SELFELICIT paper (Section 3.2):
    - Takes attention from the last token (query_position=-1) to all context tokens
    - Averages across all heads in each layer
    - Averages across the specified layer range
    - Normalizes so scores sum to 1 per layer

    Args:
        outputs: Model outputs with `attentions` attribute (tuple of tensors)
        layer_range: (start_layer, end_layer) inclusive
        context_span: (start_idx, end_idx) of context tokens in input_ids
        query_position: Position to query attention from (default: -1 = last token)

    Returns:
        attention_scores: (context_len,) numpy array of normalized attention scores
    """
    attentions = outputs.attentions  # Tuple of (batch, heads, seq, seq)
    start_l, end_l = layer_range

    # Extract attention for each evidence-reading layer
    layer_scores = []
    for l in range(start_l, end_l):
        # Shape: (batch=1, n_heads, seq_len, seq_len)
        att = attentions[l][0]  # Remove batch dim
        # Average across all heads
        att_heads_mean = att.mean(dim=0)  # (seq_len, seq_len)
        # Attention from query_position to context tokens
        att_to_context = att_heads_mean[query_position, context_span[0]:context_span[1]]
        layer_scores.append(att_to_context.detach().cpu().float().numpy())

    # Stack: (n_layers, context_len)
    layer_scores = np.stack(layer_scores, axis=0)

    # Normalize per layer (each layer sums to 1)
    layer_scores = layer_scores / layer_scores.sum(axis=1, keepdims=True)

    # Average across layers
    token_scores = layer_scores.mean(axis=0)  # (context_len,)

    return token_scores


def compute_turn_evidence_scores(
    token_scores: np.ndarray,
    turn_token_spans: List[Tuple[int, int]]
) -> np.ndarray:
    """
    Aggregate token-level attention scores to turn-level evidence scores.

    Args:
        token_scores: (context_len,) attention scores per token
        turn_token_spans: List of (start, end) token spans for each dialogue turn

    Returns:
        turn_scores: (n_turns,) evidence score per dialogue turn
    """
    turn_scores = np.array([
        token_scores[span[0]:span[1]].mean()
        for span in turn_token_spans
    ])
    return turn_scores


def select_evidence_turns(
    turn_scores: np.ndarray,
    alpha: float = DEFAULT_ALPHA
) -> np.ndarray:
    """
    Select evidence turns based on threshold: score >= alpha * max_score.

    This is the core evidence selection mechanism from SELFELICIT (Eq. 4).

    Args:
        turn_scores: (n_turns,) evidence score per turn
        alpha: Threshold multiplier (default 0.5 from paper)

    Returns:
        selected_indices: Boolean array, True for selected evidence turns
    """
    if len(turn_scores) == 0:
        return np.array([], dtype=bool)

    threshold = alpha * turn_scores.max()
    return turn_scores >= threshold


def find_turn_token_spans(
    input_ids: torch.Tensor,
    tokenizer: PreTrainedTokenizer,
    dialogue: List[dict],
    context_start: int,
    context_end: int
) -> List[Tuple[int, int]]:
    """
    Find token spans for each dialogue turn within the context portion of input_ids.

    Args:
        input_ids: Tokenized input (seq_len,) or (1, seq_len)
        tokenizer: HF tokenizer
        dialogue: List of turn dicts with 'teacher' and 'student' text
        context_start: Start token index of context within input_ids
        context_end: End token index of context within input_ids

    Returns:
        turn_spans: List of (start, end) token positions relative to full input_ids
    """
    if input_ids.dim() == 2:
        input_ids = input_ids[0]

    context_ids = input_ids[context_start:context_end]
    context_text = tokenizer.decode(context_ids)

    turn_spans = []
    for turn in dialogue:
        # Build turn text (matching format in get_dialogue_text)
        turn_lines = []
        if turn.get("teacher"):
            turn_lines.append(f"Teacher Turn {turn['turn']}: {turn['teacher']}")
        if turn.get("student"):
            turn_lines.append(f"Student Turn {turn['turn']}: {turn['student']}")

        if not turn_lines:
            continue

        turn_text = "\n".join(turn_lines)
        turn_ids = tokenizer.encode(turn_text, add_special_tokens=False)

        # Find the span in context_ids
        found = False
        for start_idx in range(len(context_ids) - len(turn_ids) + 1):
            if torch.equal(context_ids[start_idx:start_idx + len(turn_ids)],
                          torch.tensor(turn_ids, device=context_ids.device)):
                turn_spans.append((
                    context_start + start_idx,
                    context_start + start_idx + len(turn_ids)
                ))
                found = True
                break

        if not found:
            # Fuzzy match: find the best approximate location
            # This handles cases where formatting differs slightly
            turn_spans.append((context_start, context_start + 1))  # Placeholder

    return turn_spans


def get_dialogue_token_spans(
    input_ids: torch.Tensor,
    tokenizer: PreTrainedTokenizer,
    dialogue: List[dict]
) -> List[Tuple[int, int]]:
    """
    Find turn token spans by matching turn text in the decoded input.

    Uses a simpler approach: decodes the full input, then finds each turn's
    text position and converts to token indices.

    Args:
        input_ids: Tokenized input
        tokenizer: HF tokenizer
        dialogue: List of turn dicts

    Returns:
        turn_spans: List of (start, end) token positions
    """
    if input_ids.dim() == 2:
        input_ids = input_ids[0]

    full_text = tokenizer.decode(input_ids)

    turn_spans = []
    for turn in dialogue:
        turn_lines = []
        if turn.get("teacher"):
            turn_lines.append(f"Teacher Turn {turn['turn']}: {turn['teacher']}")
        if turn.get("student"):
            turn_lines.append(f"Student Turn {turn['turn']}: {turn['student']}")

        if not turn_lines:
            continue

        turn_text = "\n".join(turn_lines)

        # Find text position in decoded text
        text_pos = full_text.find(turn_text)
        if text_pos < 0:
            # Try partial match with teacher turn only
            if turn.get("teacher"):
                turn_text = f"Teacher Turn {turn['turn']}: {turn['teacher']}"
                text_pos = full_text.find(turn_text)

        if text_pos >= 0:
            # Convert char position to token position
            text_before = full_text[:text_pos]
            token_start = len(tokenizer.encode(text_before, add_special_tokens=False))
            turn_tokens = len(tokenizer.encode(turn_text, add_special_tokens=False))
            turn_spans.append((token_start, token_start + turn_tokens))
        else:
            # Could not find turn — use fallback
            turn_spans.append((0, 1))

    return turn_spans


def mark_evidence_in_dialogue_text(
    dialogue_text: str,
    dialogue: List[dict],
    evidence_mask: np.ndarray,
    marker_start: str = MARKER_IMPSTART,
    marker_end: str = MARKER_IMPEND
) -> str:
    """
    Insert evidence markers around high-importance dialogue turns.

    Args:
        dialogue_text: Original dialogue text with [BEGIN/END DIALOGUE] tags
        dialogue: List of turn dicts
        evidence_mask: Boolean array, True for evidence turns
        marker_start: Start marker string
        marker_end: End marker string

    Returns:
        modified_text: Dialogue text with evidence markers inserted
    """
    # Build the marked dialogue line by line
    lines = dialogue_text.split("\n")
    marked_lines = []
    evidence_turn_indices = set(
        i for i, is_evidence in enumerate(evidence_mask) if is_evidence
    )

    current_turn_idx = -1
    for line in lines:
        # Detect turn boundaries
        is_teacher = line.startswith("Teacher Turn ")
        is_student = line.startswith("Student Turn ")

        if is_teacher:
            # Extract turn number
            try:
                turn_num = int(line.split(":")[0].replace("Teacher Turn ", "").strip())
                current_turn_idx = turn_num - 1  # 0-indexed
            except (ValueError, IndexError):
                pass

        if (is_teacher or is_student) and current_turn_idx in evidence_turn_indices:
            marked_lines.append(f"{marker_start} {line} {marker_end}")
        else:
            marked_lines.append(line)

    return "\n".join(marked_lines)


def compute_evidence_for_sample(
    model,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    dialogue: List[dict],
    n_layers: int,
    layer_span: Tuple[float, float] = DEFAULT_LAYER_SPAN,
    alpha: float = DEFAULT_ALPHA,
    marker_start: str = MARKER_IMPSTART,
    marker_end: str = MARKER_IMPEND
) -> Tuple[str, np.ndarray, np.ndarray]:
    """
    Full SELFELICIT pipeline for a single dialogue KT sample.

    1. Run one-token forward pass to capture attention
    2. Extract attention from evidence-reading layers
    3. Aggregate to turn-level evidence scores
    4. Select evidence turns
    5. Mark evidence in prompt

    Args:
        model: The base LLM (without LoRA or with LoRA merged)
        tokenizer: HF tokenizer
        prompt: Full prompt text (system + user, before KC continuations)
        dialogue: List of annotated dialogue turns
        n_layers: Number of transformer layers
        layer_span: Fraction of layers to use as evidence-reading layers
        alpha: Threshold multiplier for evidence selection
        marker_start: Start marker for evidence
        marker_end: End marker for evidence

    Returns:
        marked_prompt: Prompt with evidence markers inserted
        turn_scores: Evidence scores per turn
        evidence_mask: Boolean mask of selected evidence turns
    """
    # Tokenize prompt
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    # Run forward pass with attention outputs
    with torch.no_grad():
        outputs = model(
            input_ids,
            output_attentions=True,
            attention_mask=torch.ones_like(input_ids),
        )

    # Get evidence-reading layer range
    n_attn_layers = len(outputs.attentions)
    layer_range = get_evidence_layer_range(n_attn_layers, layer_span)

    # Find dialogue context span via per-token decoding
    # Approach: decode each token, find [BEGIN DIALOGUE] and [END DIALOGUE]
    # in the decoded token text. This avoids the encode/decode round-trip issue.
    ids_list = input_ids[0].tolist()

    # Decode each token individually to find dialogue markers
    begin_marker = "[BEGIN DIALOGUE]"
    end_marker = "[END DIALOGUE]"

    # Build cumulative text from tokens to find marker positions
    cum_text = ""
    token_char_starts = []  # character start position of each token
    for tid in ids_list:
        token_char_starts.append(len(cum_text))
        tok_text = tokenizer.decode([tid])
        cum_text += tok_text

    # Find markers in cumulative decoded text
    b_char = cum_text.find(begin_marker)
    e_char = cum_text.find(end_marker)

    if b_char >= 0 and e_char >= 0:
        # Convert character positions to token indices
        context_start = 0
        context_end = len(ids_list)
        for ti, cs in enumerate(token_char_starts):
            if cs <= b_char:
                context_start = ti
            if cs <= e_char + len(end_marker):
                context_end = ti + 1
        # Ensure valid range
        if context_start >= context_end:
            context_start = 0
            context_end = len(ids_list)
    else:
        # Fallback: use last 70% of tokens as approximate context
        seq_len = len(ids_list)
        context_start = int(seq_len * 0.3)
        context_end = seq_len

    # Extract attention scores
    token_scores = extract_attention_scores(
        outputs, layer_range,
        (context_start, context_end),
        query_position=-1
    )

    # Find turn token spans
    turn_spans = get_dialogue_token_spans(input_ids, tokenizer, dialogue)

    # Compute turn-level evidence scores
    # For each turn span, extract the corresponding token scores
    turn_scores = np.zeros(len(turn_spans))
    for i, (t_start, t_end) in enumerate(turn_spans):
        # Map turn span to context-relative indices
        rel_start = max(0, t_start - context_start)
        rel_end = min(len(token_scores), t_end - context_start)
        if rel_start < rel_end:
            turn_scores[i] = token_scores[rel_start:rel_end].mean()

    # Select evidence turns
    evidence_mask = select_evidence_turns(turn_scores, alpha)

    # Mark evidence in prompt
    # Find the dialogue text portion and mark it
    begin_marker_text = "[BEGIN DIALOGUE]"
    dialogue_text_start = prompt.find(begin_marker_text)
    if dialogue_text_start >= 0:
        prefix = prompt[:dialogue_text_start]
        dialogue_text = prompt[dialogue_text_start:]

        # Only mark within the dialogue portion
        marked_dialogue = mark_evidence_in_dialogue_text(
            dialogue_text, dialogue, evidence_mask, marker_start, marker_end
        )
        marked_prompt = prefix + marked_dialogue
    else:
        marked_prompt = prompt

    # Clean up
    del outputs
    torch.cuda.empty_cache()

    return marked_prompt, turn_scores, evidence_mask


def precompute_evidence_annotations(
    model,
    tokenizer: PreTrainedTokenizer,
    data: 'pd.DataFrame',
    args,
    layer_span: Tuple[float, float] = DEFAULT_LAYER_SPAN,
    alpha: float = DEFAULT_ALPHA,
    debug: bool = False
) -> List[Dict]:
    """
    Pre-compute SELFELICIT evidence annotations for all samples.

    This runs the base model (with output_attentions=True) on each sample
    to identify which dialogue turns are evidence for KC prediction.

    Args:
        model: Base LLM
        tokenizer: HF tokenizer
        data: DataFrame with annotated dialogue samples
        args: Training arguments
        layer_span: Evidence-reading layer range
        alpha: Evidence selection threshold
        debug: If True, only process first 10 samples

    Returns:
        evidence_annotations: List of dicts with evidence info per sample
    """
    from dialogue_kt.kt_data_loading import apply_annotations
    from dialogue_kt.prompting import kt_system_prompt, kt_user_prompt

    model.eval()
    n_layers = model.config.num_hidden_layers
    evidence_annotations = []

    samples = data.iterrows() if not debug else list(data[:10].iterrows())

    for idx, sample in tqdm(samples, desc="Pre-computing evidence"):
        dialogue = apply_annotations(sample)
        if not dialogue:
            evidence_annotations.append(None)
            continue

        sample_evidence = {
            "dialogue_idx": idx,
            "turns": {}  # turn_idx -> evidence info
        }

        for turn in dialogue:
            if turn["correct"] is None:
                continue

            # Build the prompt for this turn (without KC continuations)
            system_content = kt_system_prompt(args)
            user_content = kt_user_prompt(sample, dialogue, turn["turn"], None, args)

            prompt = tokenizer.apply_chat_template([
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
            ], tokenize=False)

            # Compute evidence scores
            try:
                marked_prompt, turn_scores, evidence_mask = compute_evidence_for_sample(
                    model, tokenizer, prompt, dialogue,
                    n_layers, layer_span, alpha
                )

                sample_evidence["turns"][turn["turn"]] = {
                    "turn_scores": turn_scores.tolist(),
                    "evidence_mask": evidence_mask.tolist(),
                    "marked_dialogue_text": marked_prompt
                }
            except Exception as e:
                print(f"  Warning: Evidence computation failed for dialogue {idx}, turn {turn['turn']}: {e}")
                sample_evidence["turns"][turn["turn"]] = {
                    "turn_scores": [],
                    "evidence_mask": [],
                    "marked_dialogue_text": None
                }

        evidence_annotations.append(sample_evidence)

    return evidence_annotations


# ===== SELFELICIT Inference Enhancement =====

def two_pass_selfelicit_inference(
    model,
    tokenizer: PreTrainedTokenizer,
    batch: dict,
    true_token: int,
    false_token: int,
    args,
    layer_span: Tuple[float, float] = DEFAULT_LAYER_SPAN,
    alpha: float = DEFAULT_ALPHA
) -> Tuple[torch.Tensor, torch.Tensor, List[List[float]]]:
    """
    Two-pass inference with SELFELICIT evidence enhancement.

    Pass 1: Standard forward pass with attention output to identify evidence
    Pass 2: Re-run with evidence-marked prompts

    Args:
        model: The trained LMKT model
        tokenizer: HF tokenizer
        batch: Collated batch dict from LMKTCollator
        true_token, false_token: Token IDs
        args: Training args
        layer_span: Evidence-reading layer range
        alpha: Evidence selection threshold

    Returns:
        corr_probs_pass1: Predictions from standard pass
        corr_probs_pass2: Predictions from evidence-enhanced pass
        kc_probs_grouped: Per-KC probability lists
    """
    from dialogue_kt.kt_data_loading import apply_annotations
    from dialogue_kt.prompting import kt_system_prompt, kt_user_prompt
    from dialogue_kt.training import get_lmkt_loss_packed

    n_layers = model.config.num_hidden_layers
    batch_size = len(batch["meta_data"])

    # === Pass 1: Standard forward with attention ===
    attention_mask = batch["attention_mask"].clone()
    min_dtype = torch.finfo(model.dtype).min
    attention_mask[attention_mask == 0] = min_dtype
    attention_mask[attention_mask == 1] = 0
    attention_mask = attention_mask.type(model.dtype)

    with torch.no_grad():
        outputs_pass1 = model(
            input_ids=batch["input_ids"],
            attention_mask=attention_mask,
            position_ids=batch["position_ids"],
            output_attentions=True,
        )

    # Extract pass 1 predictions
    logits_pass1 = outputs_pass1.logits[
        torch.arange(batch_size).unsqueeze(1), batch["last_idxs"]
    ]
    logits_pass1 = torch.stack(
        [logits_pass1[:, :, true_token], logits_pass1[:, :, false_token]], dim=2
    )
    kc_probs_pass1 = torch.softmax(logits_pass1, dim=2)[:, :, 0]

    # Compute pass 1 correctness probabilities
    padding_val = 0 if args.agg == "mean-ar" else 1
    kc_probs_padded = torch.masked_scatter(
        kc_probs_pass1.clone(),
        batch["last_idxs"].to(device) == 0,
        torch.full_like(kc_probs_pass1, padding_val).to(device)
    )
    if args.agg == "prod":
        corr_probs_pass1 = kc_probs_padded.prod(dim=1)
    elif args.agg == "mean-ar":
        corr_probs_pass1 = kc_probs_padded.sum(dim=1) / batch["num_kcs"]
    elif args.agg == "mean-geo":
        corr_probs_pass1 = kc_probs_padded.prod(dim=1) ** (1 / batch["num_kcs"])

    # === Identify evidence from attention ===
    # For each sample in batch, find evidence turns
    evidence_masks = []
    for sample_idx, sample in enumerate(batch["meta_data"]):
        dialogue = apply_annotations(sample.get("_sample"))
        if not dialogue:
            evidence_masks.append(None)
            continue

        # Use attention from Pass 1
        # Focus on attention to dialogue context
        attentions = outputs_pass1.attentions
        n_attn_layers = len(attentions)
        layer_range = get_evidence_layer_range(n_attn_layers, layer_span)

        # Use the last KC's last token as query
        # Build prompt to find context span
        system_content = kt_system_prompt(args)
        # Find which turn this sample corresponds to
        turn_idx = None
        for turn in dialogue:
            if turn["correct"] is not None and turn["kcs"] == sample["kcs"]:
                turn_idx = turn["turn"]
                break

        if turn_idx is None:
            evidence_masks.append(None)
            continue

        user_content = kt_user_prompt(sample.get("_sample", {}), dialogue, turn_idx, None, args)
        full_prompt = tokenizer.apply_chat_template([
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ], tokenize=False)

        prompt_ids = tokenizer(full_prompt, return_tensors="pt").input_ids.to(device)
        dialogue_marker = "[BEGIN DIALOGUE]"
        full_text = tokenizer.decode(prompt_ids[0])
        ds_pos = full_text.find(dialogue_marker)

        if ds_pos >= 0:
            text_before = full_text[:ds_pos]
            context_start = len(tokenizer.encode(text_before, add_special_tokens=False))
            context_end = prompt_ids.shape[1]

            # The context in the actual input might be at different positions
            # due to padding. For simplicity, extract attention from the last
            # non-padding position to the context region.
            actual_seq_len = batch["attention_mask"][sample_idx, 0, 0].sum().item()

            # Extract and average attention
            actual_context_end = min(context_end, int(actual_seq_len))
            actual_context_start = min(context_start, actual_context_end - 1)

            # Use attention from the last position of the last KC
            last_kc_idx = batch["last_idxs"][sample_idx, -1].item()

            layer_scores = []
            for l in range(layer_range[0], min(layer_range[1], n_attn_layers)):
                att = attentions[l][0]  # (n_heads, seq, seq) — batch dim already handled
                att_mean = att.mean(dim=0)

                # Attention from last KC position to context
                if last_kc_idx < att_mean.shape[0] and actual_context_start < att_mean.shape[1]:
                    att_to_context = att_mean[last_kc_idx, actual_context_start:actual_context_end]
                    layer_scores.append(att_to_context.detach().cpu().float().numpy())

            if layer_scores:
                layer_scores = np.stack(layer_scores)
                layer_scores = layer_scores / layer_scores.sum(axis=1, keepdims=True)
                token_scores = layer_scores.mean(axis=0)

                # Find turn spans
                turn_spans = get_dialogue_token_spans(
                    batch["input_ids"][sample_idx], tokenizer, dialogue
                )

                # Compute turn scores
                turn_scores = np.zeros(len(turn_spans))
                for i, (ts, te) in enumerate(turn_spans):
                    rs = max(0, ts - actual_context_start)
                    re = min(len(token_scores), te - actual_context_start)
                    if rs < re:
                        turn_scores[i] = token_scores[rs:re].mean()

                evidence_masks.append(select_evidence_turns(turn_scores, alpha))
            else:
                evidence_masks.append(None)
        else:
            evidence_masks.append(None)

    # Clean up Pass 1 outputs
    del outputs_pass1
    torch.cuda.empty_cache()

    # === Pass 2: Would need to reconstruct marked prompts ===
    # For now, return Pass 1 results as Pass 2 placeholder
    # The full implementation would re-tokenize with marked prompts

    kc_probs_grouped = [
        probs[:num_kcs].tolist()
        for probs, num_kcs in zip(kc_probs_pass1, batch["num_kcs"])
    ]

    return corr_probs_pass1, corr_probs_pass1, kc_probs_grouped  # Pass 2 = Pass 1 for now


# ===== Attention-Weighted KC Aggregation =====

def compute_attention_weighted_aggregation(
    kc_probs: torch.Tensor,
    attention_weights: torch.Tensor,
    num_kcs: torch.Tensor,
    last_idxs: torch.Tensor,
    agg: str = "mean-ar",
    mix_lambda: float = 1.0,
) -> torch.Tensor:
    """
    Aggregate KC probabilities using attention-based weights instead of uniform weights.
    When mix_lambda < 1, interpolate between uniform aggregation and weighted aggregation.
    """
    mask = (last_idxs != 0).float()
    safe_num_kcs = num_kcs.float().clamp(min=1)

    if agg == "mean-ar":
        uniform_probs = (kc_probs * mask).sum(dim=1) / safe_num_kcs
        weighted_sum = (kc_probs * attention_weights * mask).sum(dim=1)
        weight_sum = (attention_weights * mask).sum(dim=1).clamp(min=1e-8)
        weighted_probs = weighted_sum / weight_sum
    elif agg == "prod":
        masked_probs = torch.masked_scatter(kc_probs.clone(), ~mask.bool(), torch.ones_like(kc_probs))
        uniform_probs = masked_probs.prod(dim=1)
        log_probs = torch.log(kc_probs.clamp(min=1e-8))
        weighted_log_sum = (log_probs * attention_weights * mask).sum(dim=1)
        weight_sum = (attention_weights * mask).sum(dim=1).clamp(min=1e-8)
        weighted_probs = torch.exp(weighted_log_sum / weight_sum)
    elif agg == "mean-geo":
        masked_probs = torch.masked_scatter(kc_probs.clone(), ~mask.bool(), torch.ones_like(kc_probs))
        uniform_probs = masked_probs.prod(dim=1) ** (1 / safe_num_kcs)
        log_probs = torch.log(kc_probs.clamp(min=1e-8))
        weighted_log_sum = (log_probs * attention_weights * mask).sum(dim=1)
        weighted_probs = torch.exp(weighted_log_sum / safe_num_kcs)
    else:
        raise ValueError(f"Unknown aggregation method: {agg}")

    if mix_lambda >= 1.0:
        return weighted_probs
    if mix_lambda <= 0.0:
        return uniform_probs
    return (1 - mix_lambda) * uniform_probs + mix_lambda * weighted_probs



def get_kc_attention_weights(
    outputs,
    batch: dict,
    tokenizer=None,
    layer_span: Tuple[float, float] = DEFAULT_LAYER_SPAN,
    evidence_only: bool = False,
    head_filter: bool = False,
    head_topk: int = 8,
) -> torch.Tensor:
    """
    Extract per-KC attention weights from model outputs.

    Supports:
    - legacy all-context attention
    - evidence-only attention using explicit prompt markers
    - head-filtered evidence attention using top-k evidence-focused heads
    """
    attentions = outputs.attentions
    n_layers = len(attentions)
    layer_range = get_evidence_layer_range(n_layers, layer_span)

    batch_size = batch["last_idxs"].shape[0]
    max_kcs = batch["last_idxs"].shape[1]
    weights = torch.ones(batch_size, max_kcs, device=device)

    evidence_token_mask = None
    if evidence_only and tokenizer is not None:
        evidence_token_mask = build_evidence_token_mask_from_markers(batch["input_ids"], tokenizer).to(device)

    for b in range(batch_size):
        selected_heads = None
        if head_filter:
            head_scores = []
            for l in range(layer_range[0], layer_range[1]):
                att = attentions[l][b]
                for h in range(att.shape[0]):
                    if evidence_token_mask is not None and evidence_token_mask[b].any():
                        ev_mask = evidence_token_mask[b][:att.shape[-1]].float()
                        score = (att[h] * ev_mask.unsqueeze(0)).mean().item()
                    else:
                        score = att[h].mean().item()
                    head_scores.append((l, h, score))
            head_scores = sorted(head_scores, key=lambda x: x[2], reverse=True)[:max(1, head_topk)]
            selected_heads = {}
            for l, h, _ in head_scores:
                selected_heads.setdefault(l, []).append(h)

        for k in range(max_kcs):
            if batch["last_idxs"][b, k] == 0:
                weights[b, k] = 0.0
                continue

            kc_pos = batch["last_idxs"][b, k].item()
            kc_att = 0.0
            n_used = 0
            for l in range(layer_range[0], layer_range[1]):
                att = attentions[l][b]
                if kc_pos >= att.shape[-1]:
                    continue

                if selected_heads is None:
                    att_slice = att.mean(dim=0)[kc_pos]
                else:
                    head_ids = selected_heads.get(l, [])
                    if not head_ids:
                        continue
                    att_slice = att[head_ids].mean(dim=0)[kc_pos]

                if evidence_token_mask is not None and evidence_token_mask[b].any():
                    ev_mask = evidence_token_mask[b][:att_slice.shape[0]].float()
                    denom = ev_mask.sum().clamp(min=1e-8)
                    att_val = (att_slice[:ev_mask.shape[0]] * ev_mask).sum() / denom
                else:
                    att_val = att_slice[:kc_pos + 1].mean()

                kc_att += float(att_val.item())
                n_used += 1

            weights[b, k] = kc_att / max(1, n_used)

    valid_mask = (batch["last_idxs"] != 0).float().to(device)
    weight_sums = (weights * valid_mask).sum(dim=1, keepdim=True)
    uniform = valid_mask / valid_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
    normalized = torch.where(weight_sums > 1e-8, weights / weight_sums.clamp(min=1e-8), uniform)
    return normalized * batch["num_kcs"].unsqueeze(1).float().clamp(min=1)


# ===== Hook-based attention capture (Route B) =====

def register_attention_hooks(model, n_last_layers: int = 1):
    """
    Register forward hooks on the last N self-attention layers to capture
    attention weights without requiring output_attentions=True on the full model.

    For Qwen3, attention modules are at model.model.layers[i].self_attn.
    The hook captures output[1] (attention_weights) when the output is a tuple.

    Returns (hooks_list, captured_list).
    Caller must remove hooks after forward by calling h.remove() on each.
    """
    captured = []

    def make_hook(layer_idx):
        def hook(module, input, output):
            if isinstance(output, tuple) and len(output) >= 2 and output[1] is not None:
                captured.append((layer_idx, output[1].detach()))
        return hook

    hooks = []
    total_layers = model.config.num_hidden_layers
    target_layers = range(max(0, total_layers - n_last_layers), total_layers)

    for layer_idx in target_layers:
        try:
            attn_module = model.model.layers[layer_idx].self_attn
            h = attn_module.register_forward_hook(make_hook(layer_idx))
            hooks.append(h)
        except Exception:
            pass

    return hooks, captured


def get_kc_attention_weights_from_hooks(
    captured,
    batch: dict,
    tokenizer=None,
    evidence_only: bool = False,
    head_filter: bool = False,
    head_topk: int = 8,
) -> torch.Tensor:
    """
    Same semantics as get_kc_attention_weights but accepts attention from hooks.
    captured: list of (layer_idx, attn_tensor) from register_attention_hooks.
    attn_tensor shape: (batch, n_heads, seq_len, seq_len).
    """
    if not captured:
        batch_size = batch["last_idxs"].shape[0]
        max_kcs = batch["last_idxs"].shape[1]
        return torch.ones(batch_size, max_kcs, device=device)

    batch_size = batch["last_idxs"].shape[0]
    max_kcs = batch["last_idxs"].shape[1]
    weights = torch.ones(batch_size, max_kcs, device=device)

    evidence_token_mask = None
    if evidence_only and tokenizer is not None:
        evidence_token_mask = build_evidence_token_mask_from_markers(batch["input_ids"], tokenizer).to(device)

    for b in range(batch_size):
        selected_head_ids = None
        if head_filter:
            head_scores = []
            for layer_idx, att in captured:
                att_b = att[b] if att.shape[0] > b else att[0]
                for h in range(att_b.shape[0]):
                    if evidence_token_mask is not None and evidence_token_mask[b].any():
                        ev_mask = evidence_token_mask[b][:att_b.shape[-1]].float()
                        score = (att_b[h] * ev_mask.unsqueeze(0)).mean().item()
                    else:
                        score = att_b[h].mean().item()
                    head_scores.append((layer_idx, h, score))
            head_scores = sorted(head_scores, key=lambda x: x[2], reverse=True)[:max(1, head_topk)]
            selected_head_ids = {}
            for layer_idx, h, _ in head_scores:
                selected_head_ids.setdefault(layer_idx, []).append(h)

        for k in range(max_kcs):
            if batch["last_idxs"][b, k] == 0:
                weights[b, k] = 0.0
                continue

            kc_pos = batch["last_idxs"][b, k].item()
            kc_att = 0.0
            n_used = 0

            for layer_idx, att in captured:
                att_b = att[b] if att.shape[0] > b else att[0]
                if kc_pos >= att_b.shape[-1]:
                    continue

                if selected_head_ids is None:
                    att_slice = att_b.mean(dim=0)[kc_pos]
                else:
                    head_ids = selected_head_ids.get(layer_idx, [])
                    if not head_ids:
                        continue
                    att_slice = att_b[head_ids].mean(dim=0)[kc_pos]

                if evidence_token_mask is not None and evidence_token_mask[b].any():
                    ev_mask = evidence_token_mask[b][:att_slice.shape[0]].float()
                    denom = ev_mask.sum().clamp(min=1e-8)
                    att_val = (att_slice[:ev_mask.shape[0]] * ev_mask).sum() / denom
                else:
                    att_val = att_slice[:kc_pos + 1].mean()

                kc_att += float(att_val.item())
                n_used += 1

            weights[b, k] = kc_att / max(1, n_used)

    valid_mask = (batch["last_idxs"] != 0).float().to(device)
    weight_sums = (weights * valid_mask).sum(dim=1, keepdim=True)
    uniform = valid_mask / valid_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
    normalized = torch.where(weight_sums > 1e-8, weights / weight_sums.clamp(min=1e-8), uniform)
    return normalized * batch["num_kcs"].unsqueeze(1).float().clamp(min=1)
