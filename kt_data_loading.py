from typing import Dict, List
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sentence_transformers import SentenceTransformer

from dialogue_kt.models.dkt_sem import ALT_ARCH
from dialogue_kt.prompting import kt_system_prompt, kt_user_prompt, dkt_sem_prompt
from dialogue_kt.prompting import kt_system_prompt_se, kt_user_prompt_se
from dialogue_kt.prompting import kt_user_prompt_bayes, kt_user_prompt_informativeness
from dialogue_kt.utils import device

def apply_annotations(sample: dict, apply_na: bool = True):
    dialogue = sample["dialogue"]
    anno = sample["annotation"]
    if "error" in anno:
        return None
    # Handle dialogues beginning with turn 0 (student-initiated)
    if dialogue[0]["turn"] == 0:
        anno["turn 0"] = {"correct": None, "kcs": []}
    # Copy correctness and kcs into dialogue
    for dia_turn in dialogue:
        anno_turn = anno[f"turn {dia_turn['turn']}"]
        corr = anno_turn["correct"]
        kcs = anno_turn["kcs"]
        if apply_na:
            corr = None if not kcs else corr
            kcs = [] if corr is None else kcs
        dia_turn["correct"] = dia_turn["og_correct"] = corr
        dia_turn["kcs"] = kcs
    # Use human annotation of correctness for final turn
    if dialogue[-1]["kcs"]: # Skip if no KCs for final turn since correct must be None
        if "expected_result" in sample["meta_data"]: # CoMTA
            dialogue[-1]["correct"] = sample["meta_data"]["expected_result"] == "Answer Accepted"
        elif "self_correctness" in sample["meta_data"]: # MathDial
            if dialogue[-1]["correct"] is not None: # Final turn could be closing remarks, so skip if not tagged as having correctness
                if sample["meta_data"]["self_correctness"] == "Yes":
                    dialogue[-1]["correct"] = True
                elif sample["meta_data"]["self_correctness"] == "Yes, but I had to reveal the answer":
                    dialogue[-1]["correct"] = None
                elif sample["meta_data"]["self_correctness"] == "No":
                    dialogue[-1]["correct"] = False
    return dialogue

class DatasetBase(Dataset):
    def __getitem__(self, index: int):
        return self.data[index]

    def __len__(self):
        return len(self.data)

class LMKTDatasetUnpacked(DatasetBase):
    def __init__(self, data: pd.DataFrame, tokenizer, args, skip_first_turn: bool = False):
        self.data = []
        failed = 0
        for idx, sample in data.iterrows():
            dialogue = apply_annotations(sample)
            if not dialogue:
                failed += 1
                continue
            is_first_turn = True
            for turn in dialogue:
                if turn["correct"] is None:
                    continue
                # Skip first tagged turn at test time for fairness with baselines
                if skip_first_turn and is_first_turn:
                    is_first_turn = False
                    continue
                self.data.append({
                    "dialogue_idx": idx,
                    "prompts": [
                        tokenizer.apply_chat_template([
                            {"role": "system", "content": kt_system_prompt(args)},
                            {"role": "user", "content": kt_user_prompt(sample, dialogue, turn["turn"], kc, args)},
                            {"role": "assistant", "content": f"\n"} # Newline would precede True or False prediction
                        ], tokenize=False)
                        for kc in turn["kcs"]
                    ],
                    "label": turn["correct"],
                    "kcs": turn["kcs"]
                })
        print(f"{failed} / {len(data)} dialogues failed processing")
        print(f"Number of data points: {len(self.data)}")

class LMKTCollatorUnpacked:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        all_prompts = [prompt for sample in batch for prompt in sample["prompts"]]
        prompts_tokenized = self.tokenizer(all_prompts, return_tensors="pt", padding=True).to(device)
        input_ids = prompts_tokenized.input_ids
        dialogue_ranges = []
        for seq_idx in range(input_ids.shape[0]):
            dialogue_ranges.append(_find_dialogue_token_range(input_ids[seq_idx], self.tokenizer))
        return {
            "input_ids": input_ids,
            "attention_mask": prompts_tokenized.attention_mask,
            "last_idxs": prompts_tokenized.attention_mask.sum(dim=-1) - 2, # Take index of token before eos
            "num_kcs": torch.LongTensor([len(sample["prompts"]) for sample in batch]).to(device),
            "labels": torch.Tensor([sample["label"] for sample in batch]).to(device),
            "dialogue_token_ranges": dialogue_ranges,
            "prompt_kcs": [kc for sample in batch for kc in sample["kcs"]],
            "meta_data": batch
        }

class LMKTDatasetPacked(DatasetBase):
    def _build_packed_prompt(self, tokenizer, system_content: str, user_content: str, kcs: List[str]):
        prompt = tokenizer.apply_chat_template([
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ], tokenize=False)
        kc_conts = [
            tokenizer.apply_chat_template([
                {"role": "user", "content": kc},
                {"role": "assistant", "content": f"\n"}
            ], tokenize=False)
            for kc in kcs
        ]
        kc_template = tokenizer.apply_chat_template([
            {"role": "user", "content": "KC_PLACEHOLDER"},
            {"role": "assistant", "content": f"\n"}
        ], tokenize=False)
        placeholder_idx = kc_template.find("KC_PLACEHOLDER")
        if placeholder_idx < 0:
            raise ValueError(f"Cannot find KC placeholder in chat template: {kc_template[:100]}")
        header_end_idx = kc_template.rfind("\n", 0, placeholder_idx)
        if header_end_idx < 0:
            raise ValueError(f"Cannot find user header end in chat template: {kc_template[:100]}")
        kc_conts = [" " + cont[header_end_idx + 1:] for cont in kc_conts]
        return prompt + "".join(kc_conts)

    def __init__(self, data: pd.DataFrame, tokenizer, args, skip_first_turn: bool = False):
        self.data = []
        failed = 0
        for idx, sample in data.iterrows():
            dialogue = apply_annotations(sample)
            if not dialogue:
                failed += 1
                continue
            is_first_turn = True
            for turn in dialogue:
                if turn["correct"] is None:
                    continue
                # Skip first tagged turn at test time for fairness with baselines
                if skip_first_turn and is_first_turn:
                    is_first_turn = False
                    continue
                from dialogue_kt.bayesian_epistemology import bayes_evidence_mask
                from dialogue_kt.informativeness_search import informativeness_strengths
                bayes_mode = getattr(args, "bayes_evidence", None)
                info_mode = getattr(args, "informativeness", None)
                kt_method = getattr(args, "kt_prompt_method", None) or getattr(args, "kt_method", "base")
                user_content = kt_user_prompt(sample, dialogue, turn["turn"], None, args)
                if kt_method != "base":
                    from dialogue_kt.dialogue_methods import kt_user_prompt_method
                    if kt_method == "dual_view_consistency":
                        user_content = kt_user_prompt(sample, dialogue, turn["turn"], None, args)
                        user_content_view2 = kt_user_prompt_method(sample, dialogue, turn["turn"], None, args)
                    elif kt_method in ("mil_noisy_and", "rank_auc", "focal_loss", "margin_loss"):
                        pass
                    else:
                        user_content = kt_user_prompt_method(sample, dialogue, turn["turn"], None, args)
                elif bayes_mode:
                    mask_mode = "adaptive" if bayes_mode == "adaptive_labels" else bayes_mode
                    strengths = bayes_evidence_mask(len(dialogue), turn["turn"], mask_mode)
                    user_content = kt_user_prompt_bayes(sample, dialogue, turn["turn"], None, strengths, args)
                elif info_mode:
                    strengths = informativeness_strengths(dialogue, turn["turn"], info_mode)
                    user_content = kt_user_prompt_informativeness(sample, dialogue, turn["turn"], None, strengths, args)
                prompt = self._build_packed_prompt(tokenizer, kt_system_prompt(args), user_content, turn["kcs"])
                sample_dict = {
                    "dialogue_idx": idx,
                    "prompt": prompt,
                    "label": turn["correct"],
                    "kcs": turn["kcs"]
                }
                if kt_method == "dual_view_consistency":
                    sample_dict["prompt_view2"] = self._build_packed_prompt(tokenizer, kt_system_prompt(args), user_content_view2, turn["kcs"])
                self.data.append(sample_dict)
        print(f"{failed} / {len(data)} dialogues failed processing")
        print(f"Number of data points: {len(self.data)}")


class LMKTDatasetPackedSE(LMKTDatasetPacked):
    """
    SELFELICIT-enhanced packed dataset.

    Extends LMKTDatasetPacked to mark evidence dialogue turns with
    <start_important> / <end_important> markers based on pre-computed
    SELFELICIT attention scores.

    If evidence_annotations is None or empty for a sample, falls back to
    standard prompting (no markers).
    """

    def __init__(self, data: 'pd.DataFrame', tokenizer, args,
                 skip_first_turn: bool = False,
                 evidence_annotations: dict = None):
        from dialogue_kt.prompting import kt_system_prompt_se, kt_user_prompt_se
        import numpy as np

        self.data = []
        self.evidence_annotations = evidence_annotations or {}
        self.se_mode = getattr(args, "selfelicit", None)
        self.recent_turns = getattr(args, "selfelicit_recent_turns", 2)
        failed = 0

        for idx, sample in data.iterrows():
            dialogue = apply_annotations(sample)
            if not dialogue:
                failed += 1
                continue

            is_first_turn = True
            for turn in dialogue:
                if turn["correct"] is None:
                    continue

                if skip_first_turn and is_first_turn:
                    is_first_turn = False
                    continue

                # Get evidence mask for this dialogue+turn
                evidence_mask, evidence_spans = self._get_evidence_annotations(idx, turn["turn"], len(dialogue), dialogue)

                # Use SE system prompt (with evidence instructions) if evidence available,
                # otherwise fall back to standard prompt
                has_evidence = evidence_mask is not None and evidence_mask.any()
                if has_evidence:
                    system_content = kt_system_prompt_se(args)
                else:
                    system_content = kt_system_prompt(args)

                # Create base prompt with (possibly evidence-marked) dialogue
                user_content = self._build_se_user_prompt(
                    sample, dialogue, turn["turn"], evidence_mask, evidence_spans, args
                )

                prompt = tokenizer.apply_chat_template([
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": user_content},
                ], tokenize=False)

                kc_conts = [
                    tokenizer.apply_chat_template([
                        {"role": "user", "content": kc},
                        {"role": "assistant", "content": f"\n"}
                    ], tokenize=False)
                    for kc in turn["kcs"]
                ]

                # Build a template KC conversation to detect the chat template format
                kc_template = tokenizer.apply_chat_template([
                    {"role": "user", "content": "KC_PLACEHOLDER"},
                    {"role": "assistant", "content": f"\n"}
                ], tokenize=False)

                placeholder_idx = kc_template.find("KC_PLACEHOLDER")
                if placeholder_idx < 0:
                    raise ValueError(f"Cannot find KC placeholder in chat template: {kc_template[:100]}")
                header_end_idx = kc_template.rfind("\n", 0, placeholder_idx)
                if header_end_idx < 0:
                    raise ValueError(f"Cannot find user header end in chat template: {kc_template[:100]}")

                kc_conts = [" " + cont[header_end_idx + 1:] for cont in kc_conts]
                prompt = prompt + "".join(kc_conts)

                self.data.append({
                    "dialogue_idx": idx,
                    "prompt": prompt,
                    "label": turn["correct"],
                    "kcs": turn["kcs"],
                    "turn_ordinal": turn["turn"],
                })

        print(f"{failed} / {len(data)} dialogues failed processing")
        print(f"Number of data points: {len(self.data)}")

    def _default_span_annotations(self, dialogue, evidence_mask=None):
        from dialogue_kt.selfelicit import extract_evidence_spans_from_utterance

        annotations = {}
        for i, turn in enumerate(dialogue):
            if evidence_mask is not None and i < len(evidence_mask) and not evidence_mask[i]:
                annotations[turn["turn"]] = {}
                continue
            turn_info = {}
            if turn.get("teacher"):
                turn_info["teacher"] = extract_evidence_spans_from_utterance(turn["teacher"])
            if turn.get("student"):
                turn_info["student"] = extract_evidence_spans_from_utterance(turn["student"])
            annotations[turn["turn"]] = turn_info
        return annotations

    def _get_evidence_annotations(self, dialogue_idx: int, turn_idx: int, n_turns: int, dialogue):
        """Retrieve turn-level masks plus span-level evidence annotations."""
        import numpy as np

        mask = np.zeros(n_turns, dtype=bool)
        ev_info = None
        key = f"{dialogue_idx}_{turn_idx}"

        if key in self.evidence_annotations:
            ev_info = self.evidence_annotations[key]
        else:
            legacy = self.evidence_annotations.get(dialogue_idx)
            if legacy is None:
                legacy = self.evidence_annotations.get(str(dialogue_idx))
            if isinstance(legacy, dict):
                turns = legacy.get("turns", {})
                ev_info = turns.get(turn_idx) or turns.get(str(turn_idx))

        if isinstance(ev_info, dict) and ev_info.get("evidence_mask") is not None:
            cached_mask = np.asarray(ev_info["evidence_mask"], dtype=bool)
            if len(cached_mask) > n_turns:
                cached_mask = cached_mask[:n_turns]
            elif len(cached_mask) < n_turns:
                padded = np.zeros(n_turns, dtype=bool)
                padded[:len(cached_mask)] = cached_mask
                cached_mask = padded
            mask = cached_mask

        if self.se_mode in ("hybrid_head_mixed", "hybrid_head_pure"):
            if turn_idx >= self.recent_turns:
                start = max(0, turn_idx - self.recent_turns + 1)
                mask[start:turn_idx + 1] = True
        elif self.se_mode in ("prompt", "combined"):
            if not mask.any() and turn_idx >= 5:
                evidence_window = min(2, turn_idx)
                start = max(0, turn_idx - evidence_window + 1)
                mask[start:turn_idx + 1] = True
        else:
            mask[:] = False

        span_annotations = self._default_span_annotations(dialogue, evidence_mask=mask if mask.any() else None)
        if isinstance(ev_info, dict):
            raw_spans = ev_info.get("evidence_spans") or ev_info.get("span_annotations") or {}
            if isinstance(raw_spans, dict):
                merged_spans = {}
                for turn_key, turn_spans in raw_spans.items():
                    try:
                        rendered_turn = int(turn_key)
                    except (TypeError, ValueError):
                        rendered_turn = turn_key
                    merged_spans[rendered_turn] = {
                        side: [tuple(span) for span in spans]
                        for side, spans in turn_spans.items()
                    }
                span_annotations.update(merged_spans)

        return mask, span_annotations

    def _build_se_user_prompt(self, sample, dialogue, turn_idx, evidence_mask, evidence_spans, args):
        """Build user prompt with turn-level and/or span-level evidence annotations."""
        from dialogue_kt.prompting import kt_user_prompt, kt_user_prompt_se

        has_turn_evidence = evidence_mask is not None and evidence_mask.any()
        has_span_evidence = has_turn_evidence and any(
            evidence_spans.get(turn["turn"], {}).get("teacher") or evidence_spans.get(turn["turn"], {}).get("student")
            for turn in dialogue
        ) if evidence_spans else False

        if has_turn_evidence or has_span_evidence:
            return kt_user_prompt_se(
                sample, dialogue, turn_idx, None, evidence_mask, args, evidence_spans=evidence_spans
            )
        else:
            return kt_user_prompt(sample, dialogue, turn_idx, None, args)


def get_evidence_cache_filename(args, split: str = "train"):
    """Get filename for cached evidence annotations."""
    from dialogue_kt.data_loading import get_annotated_data_filename
    base = get_annotated_data_filename(args, split)
    return base.replace(".csv", "_evidence.json")


def load_evidence_cache(args, split: str = "train") -> dict:
    """Load cached evidence annotations from JSON file."""
    import json
    import os
    cache_file = get_evidence_cache_filename(args, split)
    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            return json.load(f)
    return {}


def save_evidence_cache(evidence_dict: dict, args, split: str = "train"):
    """Save cached evidence annotations to JSON file."""
    import json

    def to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, dict):
            return {k: to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [to_serializable(v) for v in obj]
        if isinstance(obj, tuple):
            return [to_serializable(v) for v in obj]
        return obj

    cache_file = get_evidence_cache_filename(args, split)
    with open(cache_file, "w") as f:
        json.dump(to_serializable(evidence_dict), f, indent=2)
    print(f"Evidence cache saved to {cache_file}")


def _find_dialogue_token_range(input_ids: torch.Tensor, tokenizer) -> tuple:
    """
    Find the token range of the dialogue section in a tokenized prompt.

    Looks for [BEGIN DIALOGUE] and [END DIALOGUE] markers.
    Falls back to full sequence if markers not found.
    """
    begin_ids = tokenizer("[BEGIN DIALOGUE]", add_special_tokens=False).input_ids
    end_ids = tokenizer("[END DIALOGUE]", add_special_tokens=False).input_ids

    ids_list = input_ids.tolist()
    seq_len = len(ids_list)

    # Search for begin marker
    begin_idx = -1
    for i in range(seq_len - len(begin_ids) + 1):
        if ids_list[i:i + len(begin_ids)] == begin_ids:
            begin_idx = i
            break

    # Search for end marker
    end_idx = -1
    for i in range(seq_len - len(end_ids) + 1):
        if ids_list[i:i + len(end_ids)] == end_ids:
            end_idx = i + len(end_ids)
            break

    if begin_idx >= 0 and end_idx > begin_idx:
        return (begin_idx, end_idx)

    # Fallback: mask roughly the middle 60% (skip problem/solution, skip KC query)
    fallback_start = int(seq_len * 0.2)
    fallback_end = int(seq_len * 0.8)
    return (fallback_start, fallback_end)


class LMKTCollatorPacked:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def _collate_prompts(self, prompts):
        prompts_tokenized = self.tokenizer(prompts, return_tensors="pt", padding=True)
        input_ids = prompts_tokenized.input_ids.to(device)
        batch_size, max_seq_len = input_ids.shape
        eos_idxs = [
            (input_ids[seq_idx] == self.tokenizer.eos_token_id).nonzero().squeeze().cpu()
            for seq_idx in range(batch_size)
        ]
        # Create default lower triangular 3D attention mask
        attention_mask = torch.ones((max_seq_len, max_seq_len)).tril().repeat(batch_size, 1, 1)
        tril_mask = attention_mask[0].type(torch.bool)
        # Create default 2D position id matrix
        position_ids = torch.arange(max_seq_len).repeat(batch_size, 1)
        # Set attention mask and position ids for each sequence
        for seq_idx in range(batch_size):
            # Get end of context
            context_end_idx = eos_idxs[seq_idx][1]
            # Initialize to no attention to any tokens after context
            attention_mask[seq_idx, :, position_ids[seq_idx] >= context_end_idx] = 0
            # Update attention mask and position ids for each KC
            start_idx = context_end_idx + 1
            for end_idx in eos_idxs[seq_idx][3::2]:
                # Set position ids as if KC immediately followed context
                position_ids[seq_idx, start_idx : end_idx + 1] = torch.arange(context_end_idx, context_end_idx + end_idx - start_idx + 1)
                # Set KC attention mask to lower triangular to permit self-attention
                cur_tril_mask = tril_mask.clone()
                cur_tril_mask[end_idx + 1:] = False
                cur_tril_mask[:, :start_idx] = False
                attention_mask[seq_idx, cur_tril_mask] = 1
                # Go to next KC
                start_idx = end_idx + 1

        # Get index of token before eos for each KC, pad for easier loss computation
        last_idxs = pad_sequence([idxs[3::2] - 1 for idxs in eos_idxs], batch_first=True)
        kc_token_ranges = []
        for seq_idx in range(batch_size):
            seq_ranges = []
            start_idx = eos_idxs[seq_idx][1].item() + 1
            for end_idx in eos_idxs[seq_idx][3::2]:
                seq_ranges.append((start_idx, int(end_idx.item())))
                start_idx = int(end_idx.item()) + 1
            kc_token_ranges.append(seq_ranges)
        return input_ids, attention_mask.unsqueeze(1).to(device), position_ids.to(device), last_idxs, kc_token_ranges

    def __call__(self, batch):
        prompts = [sample["prompt"] for sample in batch]
        input_ids, attention_mask, position_ids, last_idxs, kc_token_ranges = self._collate_prompts(prompts)
        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "last_idxs": last_idxs,
            "num_kcs": torch.LongTensor([len(sample["kcs"]) for sample in batch]).to(device),
            "labels": torch.Tensor([sample["label"] for sample in batch]).to(device),
            "meta_data": batch
        }
        result["kc_token_ranges"] = kc_token_ranges
        # Add dialogue token ranges for LTIM
        dialogue_ranges = []
        for b in range(input_ids.shape[0]):
            dr = _find_dialogue_token_range(input_ids[b], self.tokenizer)
            dialogue_ranges.append(dr)
        result["dialogue_token_ranges"] = dialogue_ranges
        if "prompt_view2" in batch[0]:
            prompts_view2 = [sample["prompt_view2"] for sample in batch]
            input_ids2, attention_mask2, position_ids2, last_idxs2, _ = self._collate_prompts(prompts_view2)
            result.update({
                "input_ids_view2": input_ids2,
                "attention_mask_view2": attention_mask2,
                "position_ids_view2": position_ids2,
                "last_idxs_view2": last_idxs2,
            })
        return result

class DKTDataset(DatasetBase):
    def __init__(self, data: pd.DataFrame, kc_dict: Dict[str, int], kc_emb_matrix: torch.Tensor, sbert_model: SentenceTransformer):
        self.data = []
        failed = 0
        num_data_points = 0
        num_correct = 0
        for idx, sample in data.iterrows():
            dialogue = apply_annotations(sample)
            if not dialogue:
                failed += 1
                continue
            dialogue_data = {
                "labels": [], "labels_flat": [], "kc_ids": [], "kc_ids_flat": [], "turn_end_idxs": [],
                "teacher_turns": [], "student_turns": [], "kcs": [], "kc_embs": [],
                "dialogue": dialogue, "dialogue_idx": idx
            }
            for turn in dialogue:
                if turn["correct"] is None:
                    continue
                dialogue_data["labels"].append(turn["correct"])
                dialogue_data["kc_ids"].append([kc_dict[kc] for kc in turn["kcs"]])
                for kc in turn["kcs"]:
                    dialogue_data["labels_flat"].append(turn["correct"])
                    dialogue_data["kc_ids_flat"].append(kc_dict[kc])
                dialogue_data["turn_end_idxs"].append(len(dialogue_data["kc_ids_flat"]) - 1)
                dialogue_data["teacher_turns"].append(turn["teacher"])
                dialogue_data["student_turns"].append(turn["student"])
                dialogue_data["kcs"].append(turn["kcs"])
                if kc_emb_matrix is not None:
                    dialogue_data["kc_embs"].append(
                        kc_emb_matrix[dialogue_data["kc_ids"][-1]].mean(dim=0)
                    )
            # Add dialogue if at least 2 turns tagged, otherwise nothing to predict
            if len(dialogue_data["labels"]) > 1:
                self.data.append(dialogue_data)
                num_data_points += len(dialogue_data["labels"])
                num_correct += sum(dialogue_data["labels"])
        # Batch encode all dialogue text
        if sbert_model is not None:
            batch_size = 512
            if ALT_ARCH:
                seqs = [dkt_sem_prompt(tt, st, kcs, corr)
                        for dialogue in self.data
                        for tt, st, kcs, corr in zip(dialogue["teacher_turns"], dialogue["student_turns"], dialogue["kcs"], dialogue["labels"])]
                result_embs = []
                for batch_start_idx in range(0, len(seqs), batch_size):
                    batch = seqs[batch_start_idx : batch_start_idx + batch_size]
                    result_embs.append(sbert_model.encode(batch, convert_to_tensor=True))
                result_embs = torch.concat(result_embs, dim=0)
                turn_counter = 0
                for dialogue in self.data:
                    seq_len = len(dialogue["labels"])
                    dialogue["turn_embs"] = result_embs[turn_counter : turn_counter + seq_len]
                    turn_counter += seq_len
            else:
                seqs = [turn for dialogue in self.data for turn in dialogue["teacher_turns"]] + [
                        turn for dialogue in self.data for turn in dialogue["student_turns"]]
                result_embs = []
                for batch_start_idx in range(0, len(seqs), batch_size):
                    batch = seqs[batch_start_idx : batch_start_idx + batch_size]
                    result_embs.append(sbert_model.encode(batch, convert_to_tensor=True))
                result_embs = torch.concat(result_embs, dim=0)
                turn_counter = 0
                for dialogue in self.data:
                    seq_len = len(dialogue["labels"])
                    dialogue["teacher_embs"] = result_embs[turn_counter : turn_counter + seq_len]
                    stud_start = result_embs.shape[0] // 2
                    dialogue["student_embs"] = result_embs[stud_start + turn_counter : stud_start + turn_counter + seq_len]
                    turn_counter += seq_len
        self.majority_class = 1 if num_correct / num_data_points >= .5 else 0
        print(f"{failed} / {len(data)} dialogues failed processing")
        print(f"Num dialogues: {len(self.data)}, num data points: {num_data_points}, {num_correct} correct")

class DKTCollator:
    def __init__(self, flatten_kcs: bool):
        self.flatten_kcs = flatten_kcs

    def __call__(self, batch):
        labels = pad_sequence(
            [torch.LongTensor(seq["labels"]) for seq in batch],
            batch_first=True, padding_value=-100 # Pad with -100 to ignore loss on padding regions
        )
        # # Fill in KC ids, 2D matrix (length x max num KCs) per sequence
        num_kcs = pad_sequence(
            [torch.LongTensor([len(kc_ids) for kc_ids in seq["kc_ids"]]) for seq in batch],
            batch_first=True, padding_value=1 # Pad with 1 to avoid division by 0
        )
        max_num_kcs = num_kcs.max()
        kc_ids = torch.zeros((*num_kcs.shape, max_num_kcs), dtype=torch.long)
        for seq_idx, seq in enumerate(batch):
            for turn_idx, turn_kc_ids in enumerate(seq["kc_ids"]):
                kc_ids[seq_idx, turn_idx, :len(turn_kc_ids)] = torch.LongTensor(turn_kc_ids)

        result = {
            "labels": labels.to(device),
            "kc_ids": kc_ids.to(device),
            "num_kcs": num_kcs.to(device)
        }

        if self.flatten_kcs:
            # Add flattened versions of KC ids and labels for unrolled model input
            kc_ids_flat = pad_sequence([torch.LongTensor(seq["kc_ids_flat"]) for seq in batch], batch_first=True)
            labels_flat = pad_sequence([torch.LongTensor(seq["labels_flat"]) for seq in batch], batch_first=True)
            turn_end_idxs = pad_sequence([torch.LongTensor(seq["turn_end_idxs"]) for seq in batch], batch_first=True)
            result = {
                **result,
                "labels_flat": labels_flat.to(device),
                "kc_ids_flat": kc_ids_flat.to(device),
                "turn_end_idxs": turn_end_idxs.to(device)
            }

        # Add text embeddings for DKT-Sem
        if batch[0]["kc_embs"] and not ALT_ARCH:
            kc_embs = pad_sequence([torch.stack(seq["kc_embs"]) for seq in batch], batch_first=True)
            teacher_embs = pad_sequence([seq["teacher_embs"] for seq in batch], batch_first=True)
            student_embs = pad_sequence([seq["student_embs"] for seq in batch], batch_first=True)
            result = {
                **result,
                "kc_embs": kc_embs,
                "teacher_embs": teacher_embs,
                "student_embs": student_embs
            }
        elif "turn_embs" in batch[0]:
            turn_embs = pad_sequence([seq["turn_embs"] for seq in batch], batch_first=True)
            result = {
                **result,
                "turn_embs": turn_embs
            }

        return result

def get_dataloader(dataset: Dataset, collator, batch_size: int, shuffle: bool):
    return DataLoader(dataset, collate_fn=collator, batch_size=batch_size, shuffle=shuffle)
