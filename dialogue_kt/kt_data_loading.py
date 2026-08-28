from typing import Dict

import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from dialogue_kt.models.dkt_sem import ALT_ARCH
from dialogue_kt.prompting import dkt_sem_prompt, kt_system_prompt, kt_user_prompt
from dialogue_kt.utils import device


def apply_annotations(sample: dict, apply_na: bool = True):
    dialogue = sample["dialogue"]
    anno = sample["annotation"]
    if "error" in anno:
        return None
    if dialogue[0]["turn"] == 0:
        anno["turn 0"] = {"correct": None, "kcs": []}
    for dia_turn in dialogue:
        anno_turn = anno[f"turn {dia_turn['turn']}"]
        corr = anno_turn["correct"]
        kcs = anno_turn["kcs"]
        if apply_na:
            corr = None if not kcs else corr
            kcs = [] if corr is None else kcs
        dia_turn["correct"] = dia_turn["og_correct"] = corr
        dia_turn["kcs"] = kcs
    if dialogue[-1]["kcs"]:
        if "expected_result" in sample["meta_data"]:
            dialogue[-1]["correct"] = sample["meta_data"]["expected_result"] == "Answer Accepted"
        elif "self_correctness" in sample["meta_data"] and dialogue[-1]["correct"] is not None:
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
                if skip_first_turn and is_first_turn:
                    is_first_turn = False
                    continue
                self.data.append({
                    "dialogue_idx": idx,
                    "prompts": [
                        tokenizer.apply_chat_template([
                            {"role": "system", "content": kt_system_prompt(args)},
                            {"role": "user", "content": kt_user_prompt(sample, dialogue, turn["turn"], kc, args)},
                            {"role": "assistant", "content": "\n"},
                        ], tokenize=False)
                        for kc in turn["kcs"]
                    ],
                    "label": turn["correct"],
                    "kcs": turn["kcs"],
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
        dialogue_ranges = [
            _find_dialogue_token_range(input_ids[seq_idx], self.tokenizer)
            for seq_idx in range(input_ids.shape[0])
        ]
        return {
            "input_ids": input_ids,
            "attention_mask": prompts_tokenized.attention_mask,
            "last_idxs": prompts_tokenized.attention_mask.sum(dim=-1) - 2,
            "num_kcs": torch.LongTensor([len(sample["prompts"]) for sample in batch]).to(device),
            "labels": torch.Tensor([sample["label"] for sample in batch]).to(device),
            "dialogue_token_ranges": dialogue_ranges,
            "prompt_kcs": [kc for sample in batch for kc in sample["kcs"]],
            "meta_data": batch,
        }


class LMKTDatasetPacked(DatasetBase):
    def _build_packed_prompt(self, tokenizer, system_content: str, user_content: str, kcs):
        prompt = tokenizer.apply_chat_template([
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ], tokenize=False)
        kc_conts = [
            tokenizer.apply_chat_template([
                {"role": "user", "content": kc},
                {"role": "assistant", "content": "\n"},
            ], tokenize=False)
            for kc in kcs
        ]
        kc_template = tokenizer.apply_chat_template([
            {"role": "user", "content": "KC_PLACEHOLDER"},
            {"role": "assistant", "content": "\n"},
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
                if skip_first_turn and is_first_turn:
                    is_first_turn = False
                    continue
                user_content = kt_user_prompt(sample, dialogue, turn["turn"], None, args)
                self.data.append({
                    "dialogue_idx": idx,
                    "prompt": self._build_packed_prompt(
                        tokenizer,
                        kt_system_prompt(args),
                        user_content,
                        turn["kcs"],
                    ),
                    "label": turn["correct"],
                    "kcs": turn["kcs"],
                })
        print(f"{failed} / {len(data)} dialogues failed processing")
        print(f"Number of data points: {len(self.data)}")


def _find_dialogue_token_range(input_ids: torch.Tensor, tokenizer) -> tuple:
    begin_ids = tokenizer("[BEGIN DIALOGUE]", add_special_tokens=False).input_ids
    end_ids = tokenizer("[END DIALOGUE]", add_special_tokens=False).input_ids
    ids_list = input_ids.tolist()
    seq_len = len(ids_list)

    begin_idx = -1
    for i in range(seq_len - len(begin_ids) + 1):
        if ids_list[i:i + len(begin_ids)] == begin_ids:
            begin_idx = i
            break

    end_idx = -1
    for i in range(seq_len - len(end_ids) + 1):
        if ids_list[i:i + len(end_ids)] == end_ids:
            end_idx = i + len(end_ids)
            break

    if begin_idx >= 0 and end_idx > begin_idx:
        return (begin_idx, end_idx)

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
        attention_mask = torch.ones((max_seq_len, max_seq_len)).tril().repeat(batch_size, 1, 1)
        tril_mask = attention_mask[0].type(torch.bool)
        position_ids = torch.arange(max_seq_len).repeat(batch_size, 1)
        for seq_idx in range(batch_size):
            context_end_idx = eos_idxs[seq_idx][1]
            attention_mask[seq_idx, :, position_ids[seq_idx] >= context_end_idx] = 0
            start_idx = context_end_idx + 1
            for end_idx in eos_idxs[seq_idx][3::2]:
                position_ids[seq_idx, start_idx:end_idx + 1] = torch.arange(
                    context_end_idx,
                    context_end_idx + end_idx - start_idx + 1,
                )
                cur_tril_mask = tril_mask.clone()
                cur_tril_mask[end_idx + 1:] = False
                cur_tril_mask[:, :start_idx] = False
                attention_mask[seq_idx, cur_tril_mask] = 1
                start_idx = end_idx + 1

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
        dialogue_ranges = [
            _find_dialogue_token_range(input_ids[idx], self.tokenizer)
            for idx in range(input_ids.shape[0])
        ]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "last_idxs": last_idxs,
            "num_kcs": torch.LongTensor([len(sample["kcs"]) for sample in batch]).to(device),
            "labels": torch.Tensor([sample["label"] for sample in batch]).to(device),
            "meta_data": batch,
            "kc_token_ranges": kc_token_ranges,
            "dialogue_token_ranges": dialogue_ranges,
        }


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
                "labels": [],
                "labels_flat": [],
                "kc_ids": [],
                "kc_ids_flat": [],
                "turn_end_idxs": [],
                "teacher_turns": [],
                "student_turns": [],
                "kcs": [],
                "kc_embs": [],
                "dialogue": dialogue,
                "dialogue_idx": idx,
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
                    dialogue_data["kc_embs"].append(kc_emb_matrix[dialogue_data["kc_ids"][-1]].mean(dim=0))
            if len(dialogue_data["labels"]) > 1:
                self.data.append(dialogue_data)
                num_data_points += len(dialogue_data["labels"])
                num_correct += sum(dialogue_data["labels"])

        if sbert_model is not None:
            batch_size = 512
            if ALT_ARCH:
                seqs = [
                    dkt_sem_prompt(tt, st, kcs, corr)
                    for dialogue in self.data
                    for tt, st, kcs, corr in zip(
                        dialogue["teacher_turns"],
                        dialogue["student_turns"],
                        dialogue["kcs"],
                        dialogue["labels"],
                    )
                ]
                result_embs = []
                for batch_start_idx in range(0, len(seqs), batch_size):
                    batch = seqs[batch_start_idx: batch_start_idx + batch_size]
                    result_embs.append(sbert_model.encode(batch, convert_to_tensor=True))
                result_embs = torch.concat(result_embs, dim=0)
                turn_counter = 0
                for dialogue in self.data:
                    seq_len = len(dialogue["labels"])
                    dialogue["turn_embs"] = result_embs[turn_counter: turn_counter + seq_len]
                    turn_counter += seq_len
            else:
                seqs = [turn for dialogue in self.data for turn in dialogue["teacher_turns"]] + [
                    turn for dialogue in self.data for turn in dialogue["student_turns"]
                ]
                result_embs = []
                for batch_start_idx in range(0, len(seqs), batch_size):
                    batch = seqs[batch_start_idx: batch_start_idx + batch_size]
                    result_embs.append(sbert_model.encode(batch, convert_to_tensor=True))
                result_embs = torch.concat(result_embs, dim=0)
                turn_counter = 0
                for dialogue in self.data:
                    seq_len = len(dialogue["labels"])
                    dialogue["teacher_embs"] = result_embs[turn_counter: turn_counter + seq_len]
                    stud_start = result_embs.shape[0] // 2
                    dialogue["student_embs"] = result_embs[stud_start + turn_counter: stud_start + turn_counter + seq_len]
                    turn_counter += seq_len

        self.majority_class = 1 if num_correct / num_data_points >= 0.5 else 0
        print(f"{failed} / {len(data)} dialogues failed processing")
        print(f"Num dialogues: {len(self.data)}, num data points: {num_data_points}, {num_correct} correct")


class DKTCollator:
    def __init__(self, flatten_kcs: bool):
        self.flatten_kcs = flatten_kcs

    def __call__(self, batch):
        labels = pad_sequence(
            [torch.LongTensor(seq["labels"]) for seq in batch],
            batch_first=True,
            padding_value=-100,
        )
        num_kcs = pad_sequence(
            [torch.LongTensor([len(kc_ids) for kc_ids in seq["kc_ids"]]) for seq in batch],
            batch_first=True,
            padding_value=1,
        )
        max_num_kcs = num_kcs.max()
        kc_ids = torch.zeros((*num_kcs.shape, max_num_kcs), dtype=torch.long)
        for seq_idx, seq in enumerate(batch):
            for turn_idx, turn_kc_ids in enumerate(seq["kc_ids"]):
                kc_ids[seq_idx, turn_idx, :len(turn_kc_ids)] = torch.LongTensor(turn_kc_ids)

        result = {
            "labels": labels.to(device),
            "kc_ids": kc_ids.to(device),
            "num_kcs": num_kcs.to(device),
        }

        if self.flatten_kcs:
            kc_ids_flat = pad_sequence([torch.LongTensor(seq["kc_ids_flat"]) for seq in batch], batch_first=True)
            labels_flat = pad_sequence([torch.LongTensor(seq["labels_flat"]) for seq in batch], batch_first=True)
            turn_end_idxs = pad_sequence([torch.LongTensor(seq["turn_end_idxs"]) for seq in batch], batch_first=True)
            result = {
                **result,
                "labels_flat": labels_flat.to(device),
                "kc_ids_flat": kc_ids_flat.to(device),
                "turn_end_idxs": turn_end_idxs.to(device),
            }

        if batch[0]["kc_embs"] and not ALT_ARCH:
            kc_embs = pad_sequence([torch.stack(seq["kc_embs"]) for seq in batch], batch_first=True)
            teacher_embs = pad_sequence([seq["teacher_embs"] for seq in batch], batch_first=True)
            student_embs = pad_sequence([seq["student_embs"] for seq in batch], batch_first=True)
            result = {
                **result,
                "kc_embs": kc_embs,
                "teacher_embs": teacher_embs,
                "student_embs": student_embs,
            }
        elif "turn_embs" in batch[0]:
            turn_embs = pad_sequence([seq["turn_embs"] for seq in batch], batch_first=True)
            result = {
                **result,
                "turn_embs": turn_embs,
            }

        return result


def get_dataloader(dataset: Dataset, collator, batch_size: int, shuffle: bool):
    return DataLoader(dataset, collate_fn=collator, batch_size=batch_size, shuffle=shuffle)
