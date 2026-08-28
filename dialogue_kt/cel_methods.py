from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def _zero_init_linear(layer: nn.Linear):
    nn.init.zeros_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


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
        self.hidden = nn.Sequential(
            nn.Linear(hidden_dim, gate_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.out = nn.Linear(gate_hidden_dim, 1)
        _zero_init_linear(self.out)

    def forward(self, hidden_states: torch.Tensor, context_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        delta = torch.tanh(self.out(self.hidden(hidden_states)).squeeze(-1))
        if context_mask is not None:
            delta = delta * context_mask.to(dtype=delta.dtype)
        return delta


class MLPShiftSelector(nn.Module):
    def __init__(self, hidden_dim: int, gate_hidden_dim: Optional[int] = None, dropout: float = 0.1):
        super().__init__()
        gate_hidden_dim = gate_hidden_dim or max(64, hidden_dim // 4)
        self.hidden = nn.Sequential(
            nn.Linear(hidden_dim, gate_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.out = nn.Linear(gate_hidden_dim, hidden_dim)
        _zero_init_linear(self.out)

    def forward(self, hidden_states: torch.Tensor, context_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        shift = torch.tanh(self.out(self.hidden(hidden_states)))
        if context_mask is not None:
            shift = shift * context_mask.to(dtype=shift.dtype).unsqueeze(-1)
        return shift


class AdapterGateSelector(nn.Module):
    def __init__(self, hidden_dim: int, adapter_dim: Optional[int] = None, dropout: float = 0.1):
        super().__init__()
        adapter_dim = adapter_dim or max(32, hidden_dim // 8)
        self.down = nn.Linear(hidden_dim, adapter_dim)
        self.up = nn.Linear(adapter_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Linear(hidden_dim, 1)
        _zero_init_linear(self.gate)

    def forward(self, hidden_states: torch.Tensor, context_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        z = hidden_states + self.up(self.dropout(F.gelu(self.down(hidden_states))))
        delta = torch.tanh(self.gate(z).squeeze(-1))
        if context_mask is not None:
            delta = delta * context_mask.to(dtype=delta.dtype)
        return delta


class AdapterShiftSelector(nn.Module):
    def __init__(self, hidden_dim: int, adapter_dim: Optional[int] = None, dropout: float = 0.1):
        super().__init__()
        adapter_dim = adapter_dim or max(32, hidden_dim // 8)
        self.down = nn.Linear(hidden_dim, adapter_dim)
        self.up = nn.Linear(adapter_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        _zero_init_linear(self.up)

    def forward(self, hidden_states: torch.Tensor, context_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        shift = torch.tanh(self.up(self.dropout(F.gelu(self.down(hidden_states)))))
        if context_mask is not None:
            shift = shift * context_mask.to(dtype=shift.dtype).unsqueeze(-1)
        return shift


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
        self.token_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out = nn.Linear(hidden_dim, 1)
        _zero_init_linear(self.out)

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
        query = torch.tanh(self.query_proj(torch.cat([answer_state, kc_state], dim=-1)))
        token_features = torch.tanh(self.token_proj(hidden_states))
        score = self.out(token_features * query).squeeze(-1)
        delta = torch.tanh(score)
        if context_mask is not None:
            delta = delta * context_mask.to(dtype=delta.dtype)
        return delta


class TaskConditionedShiftSelector(nn.Module):
    def __init__(self, hidden_dim: int, gate_hidden_dim: Optional[int] = None, dropout: float = 0.1):
        super().__init__()
        gate_hidden_dim = gate_hidden_dim or max(64, hidden_dim // 4)
        self.query_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, gate_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(gate_hidden_dim, hidden_dim),
        )
        self.token_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out = nn.Linear(hidden_dim, hidden_dim)
        _zero_init_linear(self.out)

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
        query = torch.tanh(self.query_proj(torch.cat([answer_state, kc_state], dim=-1)))
        token_features = torch.tanh(self.token_proj(hidden_states))
        shift = torch.tanh(self.out(token_features * query))
        if context_mask is not None:
            shift = shift * context_mask.to(dtype=shift.dtype).unsqueeze(-1)
        return shift


def _masked_token_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(dtype=values.dtype)
    return (values * mask).sum() / mask.sum().clamp(min=1.0)


def build_stage2_split_mask(
    signal: torch.Tensor,
    context_mask: torch.Tensor,
    mode: str = "topk_abs",
    topk_ratio: float = 0.1,
    sigmoid_temperature: float = 5.0,
) -> torch.Tensor:
    if signal.dim() == context_mask.dim() + 1:
        split_scores = signal.float().norm(dim=-1)
    elif signal.dim() == context_mask.dim():
        split_scores = signal.float().abs()
    else:
        raise ValueError(
            "Stage 2 split signal must be token-scalar or token-vector: "
            f"signal_shape={tuple(signal.shape)}, context_shape={tuple(context_mask.shape)}"
        )

    valid = context_mask.bool()
    if mode == "sigmoid":
        if sigmoid_temperature <= 0:
            raise ValueError("cel_env_sigmoid_temperature must be positive")
        signed_score = signal.float()
        if signed_score.dim() == context_mask.dim() + 1:
            signed_score = signed_score.norm(dim=-1)
        return torch.sigmoid(sigmoid_temperature * signed_score).to(dtype=signal.dtype) * valid.to(signal.dtype)
    if mode != "topk_abs":
        raise ValueError(f"Unsupported Stage 2 split mode: {mode}")
    if not 0 < topk_ratio < 1:
        raise ValueError("cel_env_topk_ratio must be strictly between 0 and 1")

    split_mask = torch.zeros_like(split_scores)
    for batch_idx in range(split_scores.shape[0]):
        valid_idxs = valid[batch_idx].nonzero(as_tuple=False).squeeze(-1)
        valid_count = int(valid_idxs.numel())
        if valid_count == 0:
            continue
        selected_count = max(1, int(round(valid_count * topk_ratio)))
        if valid_count > 1:
            selected_count = min(selected_count, valid_count - 1)
        local_scores = split_scores[batch_idx, valid_idxs]
        selected_local = torch.topk(local_scores, k=selected_count, largest=True, sorted=False).indices
        split_mask[batch_idx, valid_idxs[selected_local]] = 1.0
    return split_mask.to(dtype=signal.dtype)


class ShuffleEnvironmentGenerator(nn.Module):
    """Deterministically permute non-rationale tokens within each dialogue context."""

    def __init__(self, seed: int = 221):
        super().__init__()
        self.seed = int(seed)
        self.register_buffer("contract_version", torch.tensor([1], dtype=torch.int64))

    def forward(self, hidden_states: torch.Tensor, non_rationale_mask: torch.Tensor) -> torch.Tensor:
        output = torch.zeros_like(hidden_states)
        for batch_idx in range(hidden_states.shape[0]):
            valid_idxs = non_rationale_mask[batch_idx].bool().nonzero(as_tuple=False).squeeze(-1)
            token_count = int(valid_idxs.numel())
            if token_count == 0:
                continue
            generator = torch.Generator(device="cpu")
            generator.manual_seed(self.seed + token_count * 1009 + batch_idx * 9176)
            permutation = torch.randperm(token_count, generator=generator).to(device=hidden_states.device)
            output[batch_idx, valid_idxs] = hidden_states[batch_idx, valid_idxs[permutation]]
        return output


class MLPEnvironmentGenerator(nn.Module):
    def __init__(self, hidden_dim: int, env_hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.down = nn.Linear(hidden_dim, env_hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.up = nn.Linear(env_hidden_dim, hidden_dim)
        _zero_init_linear(self.up)

    def forward(self, hidden_states: torch.Tensor, non_rationale_mask: torch.Tensor) -> torch.Tensor:
        output = self.up(self.dropout(F.gelu(self.down(self.input_norm(hidden_states)))))
        return output * non_rationale_mask.to(dtype=output.dtype).unsqueeze(-1)


class TransformerEnvironmentGenerator(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        env_hidden_dim: int,
        num_layers: int = 1,
        num_heads: int = 4,
        ffn_dim: Optional[int] = None,
        dropout: float = 0.1,
        output_init_std: float = 1e-3,
    ):
        super().__init__()
        if env_hidden_dim % num_heads != 0:
            raise ValueError("cel_env_hidden_dim must be divisible by cel_env_num_heads")
        if num_layers < 1:
            raise ValueError("cel_env_num_layers must be >= 1")
        if output_init_std <= 0:
            raise ValueError("cel_env_output_init_std must be positive")
        ffn_dim = ffn_dim or env_hidden_dim * 2
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.down = nn.Linear(hidden_dim, env_hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=env_hidden_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.up = nn.Linear(env_hidden_dim, hidden_dim)
        nn.init.normal_(self.up.weight, mean=0.0, std=output_init_std)
        if self.up.bias is not None:
            nn.init.zeros_(self.up.bias)

    def forward(self, hidden_states: torch.Tensor, non_rationale_mask: torch.Tensor) -> torch.Tensor:
        valid = non_rationale_mask.bool()
        safe_valid = valid.clone()
        empty_rows = ~safe_valid.any(dim=1)
        if empty_rows.any():
            safe_valid[empty_rows, 0] = True
        encoded = self.down(self.input_norm(hidden_states))
        encoded = encoded * valid.to(dtype=encoded.dtype).unsqueeze(-1)
        encoded = self.encoder(encoded, src_key_padding_mask=~safe_valid)
        output = self.up(encoded)
        output = torch.nan_to_num(output)
        return output * valid.to(dtype=output.dtype).unsqueeze(-1)


class ContextualTransformerEnvironmentGenerator(nn.Module):
    """Capacity-adequate sequence model for the target dual-path Stage 2 flow."""

    def __init__(
        self,
        hidden_dim: int,
        env_hidden_dim: int = 1024,
        num_layers: int = 4,
        num_heads: int = 8,
        ffn_dim: Optional[int] = None,
        dropout: float = 0.1,
        output_init_std: float = 1e-2,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        if env_hidden_dim % num_heads != 0:
            raise ValueError("cel_env_hidden_dim must be divisible by cel_env_num_heads")
        if num_layers < 2:
            raise ValueError("contextual Transformer B requires at least two layers")
        if output_init_std <= 0:
            raise ValueError("cel_env_output_init_std must be positive")
        ffn_dim = ffn_dim or env_hidden_dim * 4
        self.input_norm = nn.RMSNorm(hidden_dim, eps=norm_eps)
        self.input_proj = nn.Linear(hidden_dim, env_hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=env_hidden_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
        self.output_norm = nn.RMSNorm(env_hidden_dim, eps=norm_eps)
        self.output_proj = nn.Linear(env_hidden_dim, hidden_dim)
        nn.init.normal_(self.output_proj.weight, mean=0.0, std=output_init_std)
        if self.output_proj.bias is not None:
            nn.init.zeros_(self.output_proj.bias)

    def forward(self, hidden_states: torch.Tensor, context_mask: torch.Tensor) -> torch.Tensor:
        valid = context_mask.bool()
        safe_valid = valid.clone()
        empty_rows = ~safe_valid.any(dim=1)
        if empty_rows.any():
            safe_valid[empty_rows, 0] = True

        encoded = self.input_proj(self.input_norm(hidden_states))
        encoded = encoded * valid.to(dtype=encoded.dtype).unsqueeze(-1)
        padding_mask = ~safe_valid
        if self.training and torch.is_grad_enabled():
            # B can see long dialogue sequences; recompute its encoder during
            # backward so the capacity-adequate main module does not require
            # storing every attention activation.
            encoded = checkpoint(
                lambda values, mask: self.encoder(values, src_key_padding_mask=mask),
                encoded,
                padding_mask,
                use_reentrant=False,
            )
        else:
            encoded = self.encoder(encoded, src_key_padding_mask=padding_mask)
        output = self.output_proj(self.output_norm(encoded))
        output = torch.nan_to_num(output)
        return output * valid.to(dtype=output.dtype).unsqueeze(-1)


def build_environment_generator(
    mode: str,
    hidden_dim: int,
    env_hidden_dim: Optional[int] = None,
    num_layers: int = 1,
    num_heads: int = 4,
    ffn_dim: Optional[int] = None,
    dropout: float = 0.1,
    output_init_std: float = 1e-3,
    shuffle_seed: int = 221,
) -> nn.Module:
    env_hidden_dim = env_hidden_dim or max(128, hidden_dim // 4)
    if mode == "shuffle":
        return ShuffleEnvironmentGenerator(seed=shuffle_seed)
    if mode == "mlp":
        return MLPEnvironmentGenerator(hidden_dim, env_hidden_dim, dropout=dropout)
    if mode == "transformer":
        if ffn_dim is not None and ffn_dim <= 0:
            raise ValueError("cel_env_ffn_dim must be positive when provided")
        return TransformerEnvironmentGenerator(
            hidden_dim,
            env_hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            dropout=dropout,
            output_init_std=output_init_std,
        )
    if mode == "contextual_transformer":
        if ffn_dim is not None and ffn_dim <= 0:
            raise ValueError("cel_env_ffn_dim must be positive when provided")
        return ContextualTransformerEnvironmentGenerator(
            hidden_dim,
            env_hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            dropout=dropout,
            output_init_std=output_init_std,
        )
    raise ValueError(f"Unsupported Stage 2 environment mode: {mode}")


def postprocess_environment_output(
    environment: torch.Tensor,
    environment_input: torch.Tensor,
    non_rationale_mask: torch.Tensor,
    mode: str = "none",
    output_ratio: float = 0.1,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    if mode == "none":
        return environment, None
    if mode != "centered_rms":
        raise ValueError(f"Unsupported Stage 2 environment output postprocess: {mode}")
    if output_ratio <= 0:
        raise ValueError("cel_env_output_ratio must be positive")

    valid = non_rationale_mask.bool()
    valid_float = valid.to(dtype=torch.float32)
    valid_3d = valid_float.unsqueeze(-1)
    raw = environment.float()
    centered = (raw - _masked_mean(raw, valid, keepdim=True)) * valid_3d
    denominator = (valid_float.sum(dim=1) * raw.shape[-1]).clamp(min=1.0).view(-1, 1, 1)
    centered_rms = torch.sqrt(centered.square().sum(dim=(1, 2), keepdim=True) / denominator)

    input_float = environment_input.float() * valid_3d
    input_rms = torch.sqrt(input_float.square().sum(dim=(1, 2), keepdim=True) / denominator)
    output_scale = float(output_ratio) * input_rms / centered_rms.clamp(min=1e-6)
    output_scale = torch.nan_to_num(output_scale, nan=0.0, posinf=0.0, neginf=0.0).clamp(max=1e4)
    processed = centered * output_scale
    return processed.to(dtype=environment.dtype), output_scale


def _masked_tensor_rms(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_float = mask.to(dtype=torch.float32).unsqueeze(-1)
    denominator = (mask_float.sum() * values.shape[-1]).clamp(min=1.0)
    return torch.sqrt((values.float().square() * mask_float).sum() / denominator)


def compute_dual_path_statistics(
    hidden_states: torch.Tensor,
    selection_strength: torch.Tensor,
    context_mask: torch.Tensor,
    evidence_weight: torch.Tensor,
    non_evidence_weight: torch.Tensor,
    evidence_representation: torch.Tensor,
    non_evidence_representation: torch.Tensor,
    transformed_non_evidence: torch.Tensor,
    mixed_representation: torch.Tensor,
    beta: float,
    raw_transformed_non_evidence: Optional[torch.Tensor] = None,
    output_scale: Optional[torch.Tensor] = None,
) -> dict:
    valid = context_mask.bool()
    valid_strength = selection_strength.detach().float()[valid]
    valid_evidence_weight = evidence_weight.detach().float()[valid]
    valid_non_evidence_weight = non_evidence_weight.detach().float()[valid]

    if valid_strength.numel() == 0:
        valid_strength = selection_strength.detach().float().new_zeros(1)
        valid_evidence_weight = evidence_weight.detach().float().new_zeros(1)
        valid_non_evidence_weight = non_evidence_weight.detach().float().new_zeros(1)

    hidden_rms = _masked_tensor_rms(hidden_states.detach(), valid)
    evidence_rms = _masked_tensor_rms(evidence_representation.detach(), valid)
    non_evidence_rms = _masked_tensor_rms(non_evidence_representation.detach(), valid)
    transformed_rms = _masked_tensor_rms(transformed_non_evidence.detach(), valid)
    raw_transformed_rms = _masked_tensor_rms(
        raw_transformed_non_evidence.detach()
        if raw_transformed_non_evidence is not None
        else transformed_non_evidence.detach(),
        valid,
    )
    mixed_rms = _masked_tensor_rms(mixed_representation.detach(), valid)
    scaled_transformed_rms = transformed_rms * float(beta)
    complement_residual = (
        evidence_representation.detach().float()
        + non_evidence_representation.detach().float()
        - hidden_states.detach().float()
    )
    complement_error = complement_residual.abs().max()
    complement_rms_error = _masked_tensor_rms(complement_residual, valid)
    complement_relative_rms_error = complement_rms_error / hidden_rms.clamp(min=1e-8)
    weight_complement_error = (
        valid_evidence_weight + valid_non_evidence_weight - 1.0
    ).abs().max()

    evidence_pool = (
        evidence_representation.detach().float() * valid.unsqueeze(-1)
    ).sum(dim=1)
    transformed_pool = (
        transformed_non_evidence.detach().float() * valid.unsqueeze(-1)
    ).sum(dim=1)
    transformed_evidence_cosine = F.cosine_similarity(
        transformed_pool,
        evidence_pool,
        dim=-1,
        eps=1e-8,
    ).mean()

    stats = {
        "a_mean": valid_strength.mean(),
        "a_abs_mean": valid_strength.abs().mean(),
        "a_pos_frac": (valid_strength > 0).float().mean(),
        "a_neg_frac": (valid_strength < 0).float().mean(),
        "a_saturation_frac": (valid_strength.abs() >= 0.95).float().mean(),
        "a_min": valid_strength.min(),
        "a_max": valid_strength.max(),
        "w_r_mean": valid_evidence_weight.mean(),
        "w_r_q10": torch.quantile(valid_evidence_weight, 0.10),
        "w_r_q50": torch.quantile(valid_evidence_weight, 0.50),
        "w_r_q90": torch.quantile(valid_evidence_weight, 0.90),
        "w_n_mean": valid_non_evidence_weight.mean(),
        "complement_max_abs_error": complement_error,
        "complement_rms_error": complement_rms_error,
        "complement_relative_rms_error": complement_relative_rms_error,
        "weight_complement_max_abs_error": weight_complement_error,
        "h_r_rms": evidence_rms,
        "h_n_rms": non_evidence_rms,
        "h_nb_raw_rms": raw_transformed_rms,
        "h_nb_rms": transformed_rms,
        "beta_h_nb_rms": scaled_transformed_rms,
        "h_m_rms": mixed_rms,
        "h_r_to_h_rms_ratio": evidence_rms / hidden_rms.clamp(min=1e-8),
        "h_n_to_h_rms_ratio": non_evidence_rms / hidden_rms.clamp(min=1e-8),
        "h_nb_to_h_n_rms_ratio": transformed_rms / non_evidence_rms.clamp(min=1e-8),
        "beta_h_nb_to_h_r_rms_ratio": scaled_transformed_rms / evidence_rms.clamp(min=1e-8),
        "h_nb_h_r_cosine": transformed_evidence_cosine,
        "stage2_effective_beta": torch.tensor(float(beta), device=hidden_states.device),
        "dual_path_finite_frac": torch.stack([
            torch.isfinite(evidence_representation).float().mean(),
            torch.isfinite(non_evidence_representation).float().mean(),
            torch.isfinite(transformed_non_evidence).float().mean(),
            torch.isfinite(mixed_representation).float().mean(),
        ]).mean(),
    }
    if output_scale is not None:
        stats["h_nb_output_scale_mean"] = output_scale.detach().float().mean()

    # Compatibility aliases allow existing report parsers to remain useful.
    stats.update({
        "gate_mean": stats["a_mean"],
        "gate_abs_mean": stats["a_abs_mean"],
        "gate_pos_frac": stats["a_pos_frac"],
        "gate_neg_frac": stats["a_neg_frac"],
        "env_l2_mean": stats["h_nb_rms"],
        "env_scaled_l2_mean": stats["beta_h_nb_rms"],
        "env_to_rationale_l2_ratio": stats["h_nb_rms"] / evidence_rms.clamp(min=1e-8),
        "env_rationale_cosine": stats["h_nb_h_r_cosine"],
    })
    return stats


def _build_stage2_complementary_representations(
    hidden_states: torch.Tensor,
    selection_strength: torch.Tensor,
    context_mask: torch.Tensor,
) -> dict:
    if selection_strength.dim() != context_mask.dim():
        raise ValueError("Dual-path Stage 2 requires token-scalar selection strength a")
    if hidden_states.shape[:2] != selection_strength.shape:
        raise ValueError(
            "Dual-path hidden and selection shapes must agree: "
            f"hidden={tuple(hidden_states.shape)}, a={tuple(selection_strength.shape)}"
        )
    context_float = context_mask.to(dtype=hidden_states.dtype)
    a = selection_strength.to(dtype=hidden_states.dtype).clamp(min=-1.0, max=1.0)
    evidence_weight_context = (a + 1.0) * 0.5
    non_evidence_weight_context = (1.0 - a) * 0.5
    evidence_weight = context_float * evidence_weight_context + (1.0 - context_float)
    non_evidence_weight = context_float * non_evidence_weight_context

    h_r = hidden_states * evidence_weight.unsqueeze(-1)
    h_n = hidden_states * non_evidence_weight.unsqueeze(-1)
    return {
        "a": a,
        "w_r": evidence_weight,
        "w_n": non_evidence_weight,
        "h_r": h_r,
        "h_n": h_n,
    }


def build_stage2_dual_path_representations(
    hidden_states: torch.Tensor,
    selection_strength: torch.Tensor,
    environment_generator: nn.Module,
    context_mask: torch.Tensor,
    beta: float,
    environment_dtype: Optional[torch.dtype] = None,
    environment_output_postprocess: str = "centered_rms",
    environment_output_ratio: float = 1.0,
) -> dict:
    if beta < 0:
        raise ValueError("Stage 2 beta must be non-negative")

    representations = _build_stage2_complementary_representations(
        hidden_states,
        selection_strength,
        context_mask,
    )
    a = representations["a"]
    evidence_weight = representations["w_r"]
    non_evidence_weight = representations["w_n"]
    h_r = representations["h_r"]
    h_n = representations["h_n"]
    if environment_dtype is None:
        environment_dtype = _module_parameter_dtype(environment_generator, hidden_states.dtype)
    environment_input = h_n.to(dtype=environment_dtype)
    raw_h_nb = environment_generator(
        environment_input,
        context_mask.to(dtype=environment_dtype),
    )
    h_nb, output_scale = postprocess_environment_output(
        raw_h_nb,
        environment_input,
        context_mask,
        mode=environment_output_postprocess,
        output_ratio=environment_output_ratio,
    )
    h_nb_hidden_dtype = h_nb.to(dtype=hidden_states.dtype)
    h_m = h_r + float(beta) * h_nb_hidden_dtype
    stats = compute_dual_path_statistics(
        hidden_states,
        a,
        context_mask,
        evidence_weight,
        non_evidence_weight,
        h_r,
        h_n,
        h_nb,
        h_m,
        beta,
        raw_transformed_non_evidence=raw_h_nb,
        output_scale=output_scale,
    )
    representations.update({
        "raw_h_nb": raw_h_nb,
        "h_nb": h_nb,
        "h_m": h_m,
        "stats": stats,
    })
    return representations


def compute_stage2_statistics(
    hidden_states: torch.Tensor,
    signal: torch.Tensor,
    context_mask: torch.Tensor,
    split_mask: torch.Tensor,
    non_rationale_mask: torch.Tensor,
    rationale: torch.Tensor,
    environment: torch.Tensor,
    beta: float,
    raw_environment: Optional[torch.Tensor] = None,
    output_scale: Optional[torch.Tensor] = None,
    shuffle_change_frac: Optional[torch.Tensor] = None,
) -> dict:
    valid = context_mask.bool()
    valid_float = valid.to(dtype=hidden_states.dtype)
    split_float = split_mask.to(dtype=hidden_states.dtype)
    non_rationale_float = non_rationale_mask.to(dtype=hidden_states.dtype)
    signal_score = signal.detach().float().abs()
    if signal_score.dim() == context_mask.dim() + 1:
        signal_score = signal_score.norm(dim=-1)

    rationale_norm = rationale.detach().float().norm(dim=-1)
    environment_norm = environment.detach().float().norm(dim=-1)
    raw_environment_norm = (
        raw_environment.detach().float().norm(dim=-1)
        if raw_environment is not None
        else environment_norm
    )
    input_norm = (hidden_states.detach().float() * non_rationale_float.unsqueeze(-1).float()).norm(dim=-1)
    selected = valid & split_mask.bool()
    non_selected = valid & ~split_mask.bool()

    rationale_pool = (rationale.detach().float() * valid_float.unsqueeze(-1).float()).sum(dim=1)
    environment_pool = (environment.detach().float() * valid_float.unsqueeze(-1).float()).sum(dim=1)
    cosine = F.cosine_similarity(rationale_pool, environment_pool, dim=-1, eps=1e-8).mean()
    environment_input_cosine = F.cosine_similarity(
        environment.detach().float(),
        hidden_states.detach().float(),
        dim=-1,
        eps=1e-8,
    )

    rationale_l2 = _masked_token_mean(rationale_norm, valid_float.float())
    env_l2 = _masked_token_mean(environment_norm, non_rationale_float.float())
    raw_env_l2 = _masked_token_mean(raw_environment_norm, non_rationale_float.float())
    input_l2 = _masked_token_mean(input_norm, non_rationale_float.float())
    stats = {
        "env_split_frac": _masked_token_mean(split_float.float(), valid_float.float()),
        "env_non_rationale_frac": _masked_token_mean(non_rationale_float.float(), valid_float.float()),
        "env_selected_score": _masked_token_mean(signal_score, selected.float()),
        "env_non_selected_score": _masked_token_mean(signal_score, non_selected.float()),
        "rationale_l2_mean": rationale_l2,
        "env_raw_l2_mean": raw_env_l2,
        "env_l2_mean": env_l2,
        "env_scaled_l2_mean": env_l2 * float(beta),
        "env_input_l2_mean": input_l2,
        "env_to_input_l2_ratio": env_l2 / input_l2.clamp(min=1e-8),
        "env_to_rationale_l2_ratio": env_l2 / rationale_l2.clamp(min=1e-8),
        "env_pool_l2_mean": environment_pool.norm(dim=-1).mean(),
        "env_rationale_cosine": cosine,
        "env_input_cosine": _masked_token_mean(environment_input_cosine, non_rationale_float.float()),
    }
    if output_scale is not None:
        stats["env_output_scale_mean"] = output_scale.detach().float().mean()
    if shuffle_change_frac is not None:
        stats["env_shuffle_change_frac"] = shuffle_change_frac
    return stats


def apply_stage2_environment_injection(
    hidden_states: torch.Tensor,
    signal: torch.Tensor,
    environment_generator: nn.Module,
    context_mask: torch.Tensor,
    gamma: float,
    beta: float,
    split_mode: str,
    topk_ratio: float,
    sigmoid_temperature: float,
    norm=None,
    last_idxs: Optional[torch.Tensor] = None,
    application_mode: str = "token_residual",
    environment_dtype: Optional[torch.dtype] = None,
    environment_output_postprocess: str = "none",
    environment_output_ratio: float = 0.1,
):
    split_mask = build_stage2_split_mask(
        signal,
        context_mask,
        mode=split_mode,
        topk_ratio=topk_ratio,
        sigmoid_temperature=sigmoid_temperature,
    )
    non_rationale_mask = context_mask.to(dtype=split_mask.dtype) * (1.0 - split_mask)
    non_rationale_hidden = hidden_states * non_rationale_mask.to(dtype=hidden_states.dtype).unsqueeze(-1)
    if environment_dtype is None:
        environment_dtype = _module_parameter_dtype(environment_generator, hidden_states.dtype)
    environment_input = non_rationale_hidden.to(dtype=environment_dtype)
    raw_environment = environment_generator(
        environment_input,
        non_rationale_mask.to(dtype=environment_dtype),
    )
    environment, output_scale = postprocess_environment_output(
        raw_environment,
        environment_input,
        non_rationale_mask,
        mode=environment_output_postprocess,
        output_ratio=environment_output_ratio,
    )
    shuffle_change_frac = None
    if isinstance(environment_generator, ShuffleEnvironmentGenerator):
        token_changed = (raw_environment.detach().float() - environment_input.detach().float()).abs().amax(dim=-1) > 0
        shuffle_change_frac = _masked_token_mean(
            token_changed.to(dtype=hidden_states.dtype),
            non_rationale_mask.to(dtype=hidden_states.dtype),
        )
    if signal.dim() == hidden_states.dim():
        rationale = signal.to(dtype=hidden_states.dtype)
    else:
        rationale = signal.to(dtype=hidden_states.dtype).unsqueeze(-1) * hidden_states
    combined_residual = rationale + float(beta) * environment.to(dtype=hidden_states.dtype)
    injected = apply_rationale_injection(
        hidden_states,
        combined_residual,
        gamma=gamma,
        norm=norm,
        context_mask=context_mask,
        last_idxs=last_idxs,
        application_mode=application_mode,
    )
    stats = compute_stage2_statistics(
        hidden_states,
        signal,
        context_mask,
        split_mask,
        non_rationale_mask,
        rationale,
        environment,
        beta,
        raw_environment=raw_environment,
        output_scale=output_scale,
        shuffle_change_frac=shuffle_change_frac,
    )
    return injected, {
        "split_mask": split_mask,
        "non_rationale_mask": non_rationale_mask,
        "raw_environment": raw_environment,
        "environment": environment,
        "combined_residual": combined_residual,
        "stats": stats,
    }


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


def _module_parameter_dtype(module: nn.Module, fallback: torch.dtype) -> torch.dtype:
    """Return the compute dtype for a module that may be parameter-free."""
    for parameter in module.parameters():
        return parameter.dtype
    return fallback


def pool_kc_query(hidden_states: torch.Tensor, batch: dict, sample_idx: int, default_to_last_token: bool = True) -> torch.Tensor:
    span = select_kc_span_from_prompt(batch, sample_idx)
    if span is not None:
        start, end = span
        if end > start:
            return hidden_states[sample_idx, start:end].mean(dim=0)
    if default_to_last_token:
        return hidden_states[sample_idx, batch["last_idxs"][sample_idx] - 1]
    return hidden_states[sample_idx].mean(dim=0)


def _pool_rationale_summary(rationale: torch.Tensor, context_mask: torch.Tensor) -> torch.Tensor:
    mask = context_mask.to(dtype=rationale.dtype).unsqueeze(-1)
    denom = context_mask.to(dtype=rationale.dtype).sum(dim=1, keepdim=True).clamp(min=1.0)
    return (rationale * mask).sum(dim=1) / denom


def apply_rationale_injection(
    hidden_states: torch.Tensor,
    delta: torch.Tensor,
    gamma: float = 1.0,
    norm: Optional[nn.Module] = None,
    context_mask: Optional[torch.Tensor] = None,
    last_idxs: Optional[torch.Tensor] = None,
    application_mode: str = "token_residual",
):
    delta = delta.to(dtype=hidden_states.dtype)
    if delta.dim() == hidden_states.dim():
        rationale = delta
    else:
        rationale = delta.unsqueeze(-1) * hidden_states

    if application_mode == "token_residual":
        injected = hidden_states + gamma * rationale
    elif application_mode == "prediction_token_shift":
        if context_mask is None or last_idxs is None:
            raise ValueError("prediction_token_shift requires context_mask and last_idxs")
        summary = _pool_rationale_summary(rationale, context_mask)
        injected = hidden_states.clone()
        if last_idxs.dim() == 1:
            valid_last_idxs = last_idxs.clamp(min=0, max=hidden_states.shape[1] - 1)
            injected[torch.arange(hidden_states.shape[0], device=hidden_states.device), valid_last_idxs] += gamma * summary
        else:
            for batch_idx in range(hidden_states.shape[0]):
                valid_last_idxs = last_idxs[batch_idx]
                valid_last_idxs = valid_last_idxs[(valid_last_idxs > 0) & (valid_last_idxs < hidden_states.shape[1])]
                if valid_last_idxs.numel() == 0:
                    continue
                injected[batch_idx, valid_last_idxs] += gamma * summary[batch_idx].unsqueeze(0)
    else:
        raise ValueError(f"Unsupported CEL application mode: {application_mode}")

    if norm is not None:
        return norm(injected)
    return injected


def compute_signal_statistics(signal: torch.Tensor, context_mask: torch.Tensor) -> dict:
    valid = context_mask.to(dtype=signal.dtype)
    denom = valid.sum().clamp(min=1.0)
    if signal.dim() == context_mask.dim():
        masked_signal = signal * valid
        mean_signal = masked_signal.sum() / denom
        abs_mean_signal = masked_signal.abs().sum() / denom
        pos_frac = (masked_signal > 0).to(dtype=signal.dtype).sum() / denom
        neg_frac = (masked_signal < 0).to(dtype=signal.dtype).sum() / denom
        return {
            "gate_mean": mean_signal,
            "gate_abs_mean": abs_mean_signal,
            "gate_pos_frac": pos_frac,
            "gate_neg_frac": neg_frac,
        }

    masked_signal = signal * valid.unsqueeze(-1)
    shift_abs_mean = (masked_signal.abs().sum(dim=-1) * valid).sum() / denom / signal.shape[-1]
    shift_l2_mean = (masked_signal.norm(dim=-1) * valid).sum() / denom
    shift_max = (masked_signal.abs().amax(dim=-1) * valid).sum() / denom
    return {
        "shift_abs_mean": shift_abs_mean,
        "shift_l2_mean": shift_l2_mean,
        "shift_max": shift_max,
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


def _get_pre_lm_head_module(model):
    """
    Resolve the final normalization module that feeds the lm_head. For Qwen-style
    models this is typically `.model.norm`; other causal LMs may expose `.norm`
    or `.ln_f`.
    """
    candidates = [model]
    if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
        candidates.append(model.base_model.model)
    if hasattr(model, "model"):
        candidates.append(model.model)
    attr_paths = [
        ("norm",),
        ("ln_f",),
        ("final_layernorm",),
        ("model", "norm"),
        ("model", "ln_f"),
        ("transformer", "ln_f"),
    ]
    for candidate in candidates:
        for path in attr_paths:
            cur = candidate
            ok = True
            for attr in path:
                if not hasattr(cur, attr):
                    ok = False
                    break
                cur = getattr(cur, attr)
            if ok and isinstance(cur, nn.Module):
                return cur
    raise AttributeError("Unable to locate pre-lm-head normalization module for CEL injection")


def _replace_layer_output(output, injected_hidden):
    if isinstance(output, tuple):
        return (injected_hidden,) + output[1:]
    return injected_hidden


def compute_cel_selection_strength(
    hidden_states: torch.Tensor,
    batch: dict,
    selector: nn.Module,
    mode: str,
    context_mask: torch.Tensor,
    tokenizer=None,
) -> torch.Tensor:
    selector_dtype = _module_parameter_dtype(selector, torch.float32)
    if mode == "task_conditioned":
        signals = []
        for batch_idx in range(hidden_states.shape[0]):
            query_pairs = build_task_conditioned_pairs_from_batch(
                hidden_states,
                batch,
                batch_idx,
                tokenizer=tokenizer,
            )
            per_kc_signals = []
            for answer_state, kc_state in query_pairs:
                per_kc_signals.append(
                    selector(
                        hidden_states[batch_idx:batch_idx + 1].to(dtype=selector_dtype),
                        answer_state=answer_state.unsqueeze(0).to(dtype=selector_dtype),
                        kc_state=kc_state.unsqueeze(0).to(dtype=selector_dtype),
                        context_mask=context_mask[batch_idx:batch_idx + 1],
                    )
                )
            signal = torch.stack(per_kc_signals, dim=0).mean(dim=0).squeeze(0)
            signals.append(signal)
        return torch.stack(signals, dim=0)
    return selector(hidden_states.to(dtype=selector_dtype), context_mask=context_mask)


def _repeat_batch_value(value: Any, batch_size: int, repeat_count: int) -> Any:
    if isinstance(value, torch.Tensor):
        if value.dim() > 0 and value.shape[0] == batch_size:
            return torch.cat([value] * repeat_count, dim=0)
        return value
    if isinstance(value, tuple):
        return tuple(_repeat_batch_value(item, batch_size, repeat_count) for item in value)
    if isinstance(value, list):
        return [_repeat_batch_value(item, batch_size, repeat_count) for item in value]
    if isinstance(value, dict):
        return {
            key: _repeat_batch_value(item, batch_size, repeat_count)
            for key, item in value.items()
        }
    return value


def register_cel_dual_path_hook(
    model: nn.Module,
    batch: dict,
    selector: nn.Module,
    environment_generator: nn.Module,
    mode: str,
    layer_idx: int,
    beta: float,
    tokenizer=None,
    include_reversal: bool = False,
    branch_names: Optional[Sequence[str]] = None,
    environment_output_postprocess: str = "centered_rms",
    environment_output_ratio: float = 1.0,
):
    layers = _get_transformer_layers(model)
    if layer_idx < 0:
        layer_idx = len(layers) + layer_idx
    layer_idx = max(0, min(layer_idx, len(layers) - 1))
    if layer_idx != len(layers) - 1:
        raise ValueError(
            "The current dual-path batch expansion is restricted to the final Qwen block"
    )
    hook_module = layers[layer_idx]
    if branch_names is None:
        requested_branch_names = ["evidence", "mixed"]
        if include_reversal:
            requested_branch_names.append("non_evidence")
    else:
        requested_branch_names = list(branch_names)
    valid_branch_names = {"evidence", "mixed", "non_evidence"}
    if not requested_branch_names:
        raise ValueError("Dual-path hook requires at least one prediction branch")
    if len(set(requested_branch_names)) != len(requested_branch_names):
        raise ValueError(f"Dual-path hook received duplicate branches: {requested_branch_names}")
    if any(name not in valid_branch_names for name in requested_branch_names):
        raise ValueError(f"Unsupported dual-path branch request: {requested_branch_names}")
    context_mask = build_context_mask(
        batch["dialogue_token_ranges"],
        batch["input_ids"].shape[1],
        batch["input_ids"].device,
    )
    environment_dtype = _module_parameter_dtype(environment_generator, torch.float32)
    captured = {}

    def pre_hook(module, inputs, kwargs):
        if not inputs:
            raise ValueError("CEL dual-path pre-block hook received no hidden-state input")
        hidden_states = inputs[0]
        original_batch_size = hidden_states.shape[0]
        selection_strength = compute_cel_selection_strength(
            hidden_states,
            batch,
            selector,
            mode,
            context_mask,
            tokenizer=tokenizer,
        )
        if "mixed" in requested_branch_names:
            representations = build_stage2_dual_path_representations(
                hidden_states,
                selection_strength,
                environment_generator,
                context_mask,
                beta=beta,
                environment_dtype=environment_dtype,
                environment_output_postprocess=environment_output_postprocess,
                environment_output_ratio=environment_output_ratio,
            )
        else:
            # Evidence-only and reversal-only calls do not need B. Avoiding an
            # unused B forward preserves the B dropout sequence for the later
            # mixed branch and reduces the B-only warmup peak.
            representations = _build_stage2_complementary_representations(
                hidden_states,
                selection_strength,
                context_mask,
            )
        branch_hidden_by_name = {
            "evidence": representations["h_r"],
            "mixed": representations.get("h_m"),
            "non_evidence": representations["h_n"],
        }
        branch_hidden = [branch_hidden_by_name[name] for name in requested_branch_names]
        if any(hidden is None for hidden in branch_hidden):
            raise ValueError(f"Missing requested dual-path representation: {requested_branch_names}")
        repeat_count = len(branch_hidden)
        expanded_hidden = torch.cat(branch_hidden, dim=0)
        expanded_inputs = (expanded_hidden,) + tuple(
            _repeat_batch_value(item, original_batch_size, repeat_count)
            for item in inputs[1:]
        )
        expanded_kwargs = {
            key: _repeat_batch_value(value, original_batch_size, repeat_count)
            for key, value in kwargs.items()
        }

        captured["a"] = selection_strength.detach()
        captured["gate"] = selection_strength.detach()
        captured["gate_stats"] = compute_signal_statistics(
            selection_strength.detach(),
            context_mask.detach(),
        )
        captured["stage2_stats"] = {
            key: value.detach() if isinstance(value, torch.Tensor) else value
            for key, value in representations.get("stats", {}).items()
        }
        captured["branch_names"] = tuple(requested_branch_names)
        captured["original_batch_size"] = original_batch_size
        return expanded_inputs, expanded_kwargs

    handle = hook_module.register_forward_pre_hook(pre_hook, with_kwargs=True)
    return handle, captured


def register_cel_injection_hook(
    model: nn.Module,
    batch: dict,
    selector: nn.Module,
    mode: str,
    layer_idx: int,
    hook_site: str = "last_block",
    hook_timing: str = "post_block",
    gamma: float = 1.0,
    use_norm: bool = True,
    tokenizer=None,
    application_mode: str = "token_residual",
    environment_generator: Optional[nn.Module] = None,
    environment_beta: float = 0.1,
    environment_split_mode: str = "topk_abs",
    environment_topk_ratio: float = 0.1,
    environment_sigmoid_temperature: float = 5.0,
    environment_output_postprocess: str = "none",
    environment_output_ratio: float = 0.1,
):
    if hook_site == "last_block":
        layers = _get_transformer_layers(model)
        if layer_idx < 0:
            layer_idx = len(layers) + layer_idx
        layer_idx = max(0, min(layer_idx, len(layers) - 1))
        hook_module = layers[layer_idx]
    elif hook_site == "pre_lm_head":
        hook_module = _get_pre_lm_head_module(model)
    else:
        raise ValueError(f"Unsupported CEL hook site: {hook_site}")
    norm = rms_norm_tensor if use_norm else None
    context_mask = build_context_mask(batch["dialogue_token_ranges"], batch["input_ids"].shape[1], batch["input_ids"].device)
    captured = {}
    environment_dtype = (
        _module_parameter_dtype(environment_generator, torch.float32)
        if environment_generator is not None
        else None
    )

    def inject_hidden(hidden_states):
        signal = compute_cel_selection_strength(
            hidden_states,
            batch,
            selector,
            mode,
            context_mask,
            tokenizer=tokenizer,
        )
        if environment_generator is not None:
            injected, stage2_details = apply_stage2_environment_injection(
                hidden_states,
                signal,
                environment_generator,
                context_mask,
                gamma=gamma,
                beta=environment_beta,
                split_mode=environment_split_mode,
                topk_ratio=environment_topk_ratio,
                sigmoid_temperature=environment_sigmoid_temperature,
                norm=norm,
                last_idxs=batch["last_idxs"],
                application_mode=application_mode,
                environment_dtype=environment_dtype,
                environment_output_postprocess=environment_output_postprocess,
                environment_output_ratio=environment_output_ratio,
            )
            captured["stage2_stats"] = {
                key: value.detach() if isinstance(value, torch.Tensor) else value
                for key, value in stage2_details["stats"].items()
            }
        else:
            injected = apply_rationale_injection(
                hidden_states,
                signal,
                gamma=gamma,
                norm=norm,
                context_mask=context_mask,
                last_idxs=batch["last_idxs"],
                application_mode=application_mode,
            )
        captured["gate"] = signal.detach()
        captured["gate_stats"] = compute_signal_statistics(signal.detach(), context_mask.detach())
        if application_mode == "prediction_token_shift":
            delta = signal.detach().to(dtype=hidden_states.dtype)
            if delta.dim() == hidden_states.dim():
                rationale = delta
            else:
                rationale = delta.unsqueeze(-1) * hidden_states.detach()
            pred_shift = _pool_rationale_summary(rationale, context_mask.detach())
            captured["pred_shift_l2_mean"] = pred_shift.norm(dim=-1).mean().detach()
        captured["injected_mean"] = injected.mean().detach()
        return injected

    def forward_hook(module, inputs, output):
        hidden_states = output[0] if isinstance(output, tuple) else output
        injected = inject_hidden(hidden_states)
        return _replace_layer_output(output, injected)

    def pre_hook(module, inputs):
        if not inputs:
            raise ValueError("CEL pre-block hook received no positional hidden-state input")
        hidden_states = inputs[0]
        injected = inject_hidden(hidden_states)
        return (injected,) + tuple(inputs[1:])

    if hook_site != "last_block" and hook_timing != "post_block":
        raise ValueError("cel_hook_timing=pre_block is only supported with cel_hook_site=last_block")
    if hook_timing == "pre_block":
        handle = hook_module.register_forward_pre_hook(pre_hook)
    elif hook_timing == "post_block":
        handle = hook_module.register_forward_hook(forward_hook)
    else:
        raise ValueError(f"Unsupported CEL hook timing: {hook_timing}")
    return handle, captured
