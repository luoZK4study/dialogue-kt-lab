from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _masked_mean(x: torch.Tensor, mask: torch.Tensor, dim: int = 1, keepdim: bool = False) -> torch.Tensor:
    mask = mask.to(dtype=x.dtype)
    denom = mask.sum(dim=dim, keepdim=True).clamp(min=1.0)
    out = (x * mask.unsqueeze(-1)).sum(dim=dim, keepdim=True) / denom.unsqueeze(-1)
    if not keepdim:
        out = out.squeeze(dim)
    return out


def _masked_max(x: torch.Tensor, mask: torch.Tensor, dim: int = 1) -> torch.Tensor:
    neg_inf = torch.finfo(x.dtype).min
    masked = x.masked_fill(~mask.bool().unsqueeze(-1), neg_inf)
    return masked.max(dim=dim).values


def build_context_mask(dialogue_ranges: Sequence[Tuple[int, int]], seq_len: int, device) -> torch.Tensor:
    mask = torch.zeros(len(dialogue_ranges), seq_len, dtype=torch.bool, device=device)
    for i, (start, end) in enumerate(dialogue_ranges):
        start = max(0, min(start, seq_len))
        end = max(start, min(end, seq_len))
        mask[i, start:end] = True
    return mask


def find_subsequence_positions(sequence: Sequence[int], pattern: Sequence[int]) -> List[Tuple[int, int]]:
    if not sequence or not pattern or len(pattern) > len(sequence):
        return []
    positions = []
    plen = len(pattern)
    for i in range(len(sequence) - plen + 1):
        if list(sequence[i:i + plen]) == list(pattern):
            positions.append((i, i + plen))
    return positions


def find_token_span(input_ids: torch.Tensor, tokenizer, needle_text: str) -> Optional[Tuple[int, int]]:
    needle_ids = tokenizer(needle_text, add_special_tokens=False).input_ids
    if not needle_ids:
        return None
    seq = input_ids.tolist()
    spans = find_subsequence_positions(seq, needle_ids)
    return spans[0] if spans else None


def build_query_from_spans(hidden_states: torch.Tensor, spans: Sequence[Tuple[int, int]]) -> torch.Tensor:
    valid = []
    for start, end in spans:
        start = max(0, min(start, hidden_states.shape[0]))
        end = max(start, min(end, hidden_states.shape[0]))
        if end > start:
            valid.append(hidden_states[start:end].mean(dim=0))
    if valid:
        return torch.stack(valid, dim=0).mean(dim=0)
    return hidden_states.mean(dim=0)


def rms_norm_tensor(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    scale = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True).clamp(min=eps))
    return x * scale


class MLPGateSelector(nn.Module):
    def __init__(self, hidden_dim: int, gate_hidden_dim: Optional[int] = None, dropout: float = 0.1):
        super().__init__()
        gate_hidden_dim = gate_hidden_dim or max(64, hidden_dim // 4)
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, gate_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(gate_hidden_dim, 1),
        )

    def forward(self, hidden_states: torch.Tensor, context_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        gate = torch.sigmoid(self.net(hidden_states).squeeze(-1))
        if context_mask is not None:
            gate = gate * context_mask.to(dtype=gate.dtype)
        return gate


class AdapterGateSelector(nn.Module):
    def __init__(self, hidden_dim: int, adapter_dim: Optional[int] = None, dropout: float = 0.1):
        super().__init__()
        adapter_dim = adapter_dim or max(32, hidden_dim // 8)
        self.down = nn.Linear(hidden_dim, adapter_dim)
        self.up = nn.Linear(adapter_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Linear(hidden_dim, 1)

    def forward(self, hidden_states: torch.Tensor, context_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        z = hidden_states + self.up(self.dropout(F.gelu(self.down(hidden_states))))
        gate = torch.sigmoid(self.gate(z).squeeze(-1))
        if context_mask is not None:
            gate = gate * context_mask.to(dtype=gate.dtype)
        return gate


class TaskConditionedSelector(nn.Module):
    def __init__(self, hidden_dim: int, gate_hidden_dim: Optional[int] = None, dropout: float = 0.1):
        super().__init__()
        gate_hidden_dim = gate_hidden_dim or max(64, hidden_dim // 4)
        self.query_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, gate_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(gate_hidden_dim, hidden_dim),
        )
        self.score = nn.Linear(hidden_dim, 1, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        answer_state: torch.Tensor,
        kc_state: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if answer_state.dim() == 1:
            answer_state = answer_state.unsqueeze(0)
        if kc_state.dim() == 1:
            kc_state = kc_state.unsqueeze(0)
        if answer_state.dim() == 2:
            answer_state = answer_state.unsqueeze(1).expand(-1, hidden_states.shape[1], -1)
        if kc_state.dim() == 2:
            kc_state = kc_state.unsqueeze(1).expand(-1, hidden_states.shape[1], -1)
        z = self.query_proj(torch.cat([answer_state, kc_state], dim=-1))
        gate = torch.sigmoid(self.score(torch.tanh(z)).squeeze(-1))
        if context_mask is not None:
            gate = gate * context_mask.to(dtype=gate.dtype)
        return gate


def gather_dialogue_and_kc_spans(batch: dict, device) -> Tuple[torch.Tensor, torch.Tensor]:
    dialogue_ranges = batch.get("dialogue_token_ranges")
    if not dialogue_ranges:
        raise ValueError("dialogue_token_ranges is required for CEL stage 1")
    context_mask = build_context_mask(dialogue_ranges, batch["input_ids"].shape[1], device)
    return context_mask, context_mask


def select_kc_span_from_prompt(batch: dict, sample_idx: int) -> Optional[Tuple[int, int]]:
    if "kc_token_ranges" in batch and batch["kc_token_ranges"]:
        kc_ranges = batch["kc_token_ranges"]
        if sample_idx < len(kc_ranges):
            return kc_ranges[sample_idx]
    return None


def build_kc_query_from_batch(hidden_states: torch.Tensor, batch: dict, sample_idx: int, tokenizer=None) -> torch.Tensor:
    spans = []
    kc_ranges = batch.get("kc_token_ranges") or []
    if sample_idx < len(kc_ranges):
        spans = kc_ranges[sample_idx]
    elif tokenizer is not None and batch.get("prompt_kcs") is not None:
        kc_text = batch["prompt_kcs"][sample_idx]
        kc_span = find_token_span(batch["input_ids"][sample_idx], tokenizer, kc_text)
        if kc_span is not None:
            spans = [kc_span]
    if spans:
        return build_query_from_spans(hidden_states[sample_idx], spans)
    return hidden_states[sample_idx, batch["last_idxs"][sample_idx].clamp(min=0)]


def build_task_conditioned_pairs_from_batch(hidden_states: torch.Tensor, batch: dict, sample_idx: int, tokenizer=None) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    pairs = []
    kc_ranges = batch.get("kc_token_ranges") or []
    last_idxs = batch["last_idxs"][sample_idx]
    if sample_idx < len(kc_ranges) and kc_ranges[sample_idx]:
        for k_idx, span in enumerate(kc_ranges[sample_idx]):
            if last_idxs.dim() == 0:
                idx = int(last_idxs.item())
            else:
                if k_idx >= last_idxs.shape[0]:
                    continue
                idx = int(last_idxs[k_idx].item())
            if idx <= 0:
                continue
            answer_state = hidden_states[sample_idx, idx]
            kc_state = build_query_from_spans(hidden_states[sample_idx], [span])
            pairs.append((answer_state, kc_state))
    elif tokenizer is not None and batch.get("prompt_kcs") is not None:
        kc_text = batch["prompt_kcs"][sample_idx]
        kc_span = find_token_span(batch["input_ids"][sample_idx], tokenizer, kc_text)
        if kc_span is not None:
            idx = int(batch["last_idxs"][sample_idx].clamp(min=0).item())
            answer_state = hidden_states[sample_idx, idx]
            kc_state = build_query_from_spans(hidden_states[sample_idx], [kc_span])
            pairs.append((answer_state, kc_state))
    if not pairs:
        idx = int(batch["last_idxs"][sample_idx].clamp(min=0).item())
        state = hidden_states[sample_idx, idx]
        pairs.append((state, state))
    return pairs


def pool_kc_query(hidden_states: torch.Tensor, batch: dict, sample_idx: int, default_to_last_token: bool = True) -> torch.Tensor:
    span = select_kc_span_from_prompt(batch, sample_idx)
    if span is not None:
        start, end = span
        if end > start:
            return hidden_states[sample_idx, start:end].mean(dim=0)
    if default_to_last_token:
        return hidden_states[sample_idx, batch["last_idxs"][sample_idx] - 1]
    return hidden_states[sample_idx].mean(dim=0)


def apply_rationale_injection(hidden_states: torch.Tensor, gate: torch.Tensor, gamma: float = 1.0, norm: Optional[nn.Module] = None) -> torch.Tensor:
    gate = gate.to(dtype=hidden_states.dtype)
    rationale = gate.unsqueeze(-1) * hidden_states
    injected = hidden_states + gamma * rationale
    if norm is not None:
        return norm(injected)
    return injected


def compute_gate_statistics(gate: torch.Tensor, context_mask: torch.Tensor) -> dict:
    valid = context_mask.to(dtype=gate.dtype)
    denom = valid.sum().clamp(min=1.0)
    masked_gate = gate * valid
    mean_gate = masked_gate.sum() / denom
    entropy = -(masked_gate.clamp(1e-6, 1 - 1e-6) * masked_gate.clamp(1e-6, 1 - 1e-6).log()).sum() / denom
    return {
        "gate_mean": mean_gate,
        "gate_entropy": entropy,
    }


def _get_base_model(model):
    if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
        return model.base_model.model
    if hasattr(model, "model"):
        return model.model
    return model


def _get_transformer_layers(model):
    """
    Resolve the transformer block list across plain HF models and PEFT wrappers.
    Qwen-style models may expose layers on `.model.layers` instead of `.layers`.
    """
    candidates = [model]
    if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
        candidates.append(model.base_model.model)
    if hasattr(model, "model"):
        candidates.append(model.model)
    for candidate in candidates:
        if hasattr(candidate, "layers"):
            return candidate.layers
        if hasattr(candidate, "model") and hasattr(candidate.model, "layers"):
            return candidate.model.layers
    raise AttributeError("Unable to locate transformer layers for CEL injection")


def _replace_layer_output(output, injected_hidden):
    if isinstance(output, tuple):
        return (injected_hidden,) + output[1:]
    return injected_hidden


def register_cel_injection_hook(
    model: nn.Module,
    batch: dict,
    selector: nn.Module,
    mode: str,
    layer_idx: int,
    gamma: float = 1.0,
    use_norm: bool = True,
    tokenizer=None,
):
    layers = _get_transformer_layers(model)
    if layer_idx < 0:
        layer_idx = len(layers) + layer_idx
    layer_idx = max(0, min(layer_idx, len(layers) - 1))
    norm = rms_norm_tensor if use_norm else None
    context_mask = build_context_mask(batch["dialogue_token_ranges"], batch["input_ids"].shape[1], batch["input_ids"].device)
    captured = {}

    def hook(module, inputs, output):
        hidden_states = output[0] if isinstance(output, tuple) else output
        if mode == "task_conditioned":
            gates = []
            for b in range(hidden_states.shape[0]):
                query_pairs = build_task_conditioned_pairs_from_batch(hidden_states, batch, b, tokenizer=tokenizer)
                per_kc_gates = []
                for answer_state, kc_state in query_pairs:
                    per_kc_gates.append(
                        selector(
                            hidden_states[b:b + 1],
                            answer_state=answer_state.unsqueeze(0),
                            kc_state=kc_state.unsqueeze(0),
                            context_mask=context_mask[b:b + 1],
                        )
                    )
                gate_b = torch.stack(per_kc_gates, dim=0).mean(dim=0).squeeze(0)
                gates.append(gate_b)
            gate = torch.stack(gates, dim=0)
        else:
            gate = selector(hidden_states, context_mask=context_mask)
        injected = apply_rationale_injection(hidden_states, gate, gamma=gamma, norm=norm)
        captured["gate"] = gate.detach()
        captured["injected_mean"] = injected.mean().detach()
        return _replace_layer_output(output, injected)

    handle = layers[layer_idx].register_forward_hook(hook)
    return handle, captured
