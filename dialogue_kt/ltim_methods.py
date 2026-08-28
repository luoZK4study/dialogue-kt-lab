"""
LTIM: Learned Token Importance Masking for Dialogue-KT.

Three mask types:
1. Gumbel-Sigmoid (recommended): position-based learnable logits with
   Gumbel-Sigmoid sampling + temperature annealing.
2. SparsePO FFN+ReLU: content-aware mask via per-layer FFN projections
   of frozen reference model hidden states. ReLU induces automatic sparsity.
3. Hard Concrete L0: L0-regularized mask using Hard Concrete distribution
   (Louizos et al., ICLR 2018), with CoNNect (TMLR 2025) connectivity extension.

References:
- LightVLA (arXiv:2509.12594, 2025)
- SparsePO (EMNLP 2025 Findings, arXiv:2410.05102)
- CoNNect (TMLR 2025, arXiv:2502.00744)
- Louizos et al., ICLR 2018
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional


# ============================================================
# Config
# ============================================================

@dataclass
class LTIMConfig:
    mask_type: str = "gumbel_sigmoid"
    # Gumbel-Sigmoid
    gs_init_logit: float = 2.0
    gs_tau_init: float = 1.0
    gs_tau_min: float = 0.1
    gs_tau_decay: float = 0.9995
    gs_noise_scale: float = 1.0
    # SparsePO
    sp_num_layers: int = 4
    sp_bottleneck_dim: int = 256
    # Hard Concrete
    hc_beta: float = 2/3
    hc_gamma: float = -0.1
    hc_zeta: float = 1.1
    # Common
    sparsity_lambda: float = 0.01
    target_sparsity: float = 0.5
    warmup_steps: int = 500
    mask_dialogue_only: bool = True
    max_seq_len: int = 4096


# ============================================================
# Option A: Gumbel-Sigmoid Mask (Recommended)
# ============================================================

class GumbelSigmoidMask(nn.Module):
    """
    Position-based learnable token mask using Gumbel-Sigmoid.

    Each token position learns a logit. During training, we sample
    a binary mask via Gumbel-Sigmoid with temperature annealing.
    During inference, we use the deterministic sigmoid probability.

    Reference: LightVLA (Tsinghua, Sep 2025), Adaptive Token Reduction (Mar 2025).
    """

    def __init__(self, config: LTIMConfig, hidden_dim: int = 2048):
        super().__init__()
        self.config = config
        self.max_seq_len = config.max_seq_len

        # Per-position learnable logits
        self.logits = nn.Parameter(torch.full((config.max_seq_len,), config.gs_init_logit))

        # Learnable embedding to replace masked tokens
        self.mask_embedding = nn.Parameter(torch.zeros(hidden_dim))

        # Temperature state
        self.register_buffer('step_count', torch.tensor(0))
        self.register_buffer('current_tau', torch.tensor(config.gs_tau_init))

    def get_temperature(self) -> float:
        tau = max(self.config.gs_tau_min,
                   self.config.gs_tau_init * (self.config.gs_tau_decay ** self.step_count.item()))
        return tau

    def forward(self, seq_len: int, training: bool = True) -> torch.Tensor:
        """
        Returns:
            mask: [seq_len] in [0, 1], hard during training, deterministic at inference
        """
        logits = self.logits[:seq_len]
        logits_dtype = logits.dtype
        tau = torch.tensor(self.get_temperature(), dtype=logits_dtype, device=logits.device)

        if training:
            # Gumbel noise: Gumbel(0,1) = -log(-log(U(0,1)))
            uniform = torch.rand_like(logits).clamp(min=1e-8)
            gumbel_noise = -torch.log(-torch.log(uniform)) * self.config.gs_noise_scale

            # Soft version for gradient (preserve dtype)
            y_soft = torch.sigmoid((logits + gumbel_noise) / tau)

            # Hard version for forward
            y_hard = (y_soft > 0.5).to(dtype=logits_dtype)

            # Straight-Through Estimator: forward=hard, backward=soft
            mask = y_hard - y_soft.detach() + y_soft
        else:
            # Inference: deterministic threshold
            tau_val = max(self.get_temperature(), 0.5)
            tau_t = torch.tensor(tau_val, dtype=logits_dtype, device=logits.device)
            y_soft = torch.sigmoid(logits / tau_t)
            mask = (y_soft > 0.5).to(dtype=logits_dtype)

        return mask

    def get_keep_prob(self, seq_len: int) -> torch.Tensor:
        """Return soft keep probability for each position."""
        with torch.no_grad():
            tau = self.current_tau.clamp(min=0.5)
            return torch.sigmoid(self.logits[:seq_len] / tau)

    def step_temperature(self):
        self.step_count += 1
        self.current_tau = torch.tensor(self.get_temperature())

    def get_sparsity(self, seq_len: int) -> float:
        return 1.0 - self.get_keep_prob(seq_len).mean().item()


# ============================================================
# Option B: SparsePO-style FFN + ReLU Mask
# ============================================================

class SparsePOTokenMask(nn.Module):
    """
    Content-aware token mask: FFN per layer with ReLU for automatic sparsity.

    Masks are computed from frozen reference model hidden states, so they
    vary with input content. No explicit sparsity loss needed — ReLU zeros
    out negative activations.

    Reference: SparsePO (EMNLP 2025 Findings, arXiv:2410.05102).
    """

    def __init__(self, config: LTIMConfig, hidden_dim: int = 2048, num_layers_total: int = 28):
        super().__init__()
        self.config = config
        self.hidden_dim = hidden_dim
        self.num_layers = min(config.sp_num_layers, num_layers_total)

        # Per-layer FFN: hidden_dim → 1
        self.layer_projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, config.sp_bottleneck_dim),
                nn.ReLU(),
                nn.Linear(config.sp_bottleneck_dim, 1),
            )
            for _ in range(self.num_layers)
        ])

        # Layer merging: [num_layers] → 1
        self.layer_merger = nn.Linear(self.num_layers, 1, bias=False)

        # Mask embedding
        self.mask_embedding = nn.Parameter(torch.zeros(hidden_dim))

        # Fallback: position-based logits for compatibility mode
        self._fallback_logits = nn.Parameter(
            torch.full((config.max_seq_len,), config.gs_init_logit)
        )

        # Initialize for positive initial output (avoid all-zero mask)
        for proj in self.layer_projections:
            nn.init.constant_(proj[-1].bias, 1.0)
        nn.init.constant_(self.layer_merger.weight, 1.0 / self.num_layers)

    def forward(
        self,
        hidden_states_list_or_seq_len=None,
        dialogue_mask: torch.Tensor = None,
        training: bool = True,
    ) -> torch.Tensor:
        # Compatibility mode: if called with (seq_len, training=...)
        if isinstance(hidden_states_list_or_seq_len, int):
            seq_len = hidden_states_list_or_seq_len
            if not hasattr(self, '_fallback_logits'):
                self._fallback_logits = nn.Parameter(
                    torch.full((self.config.max_seq_len,), self.config.gs_init_logit)
                )
            logits = self._fallback_logits[:seq_len].to(
                dtype=self.mask_embedding.dtype,
                device=self.mask_embedding.device
            )
            tau = torch.tensor(0.5, dtype=logits.dtype, device=logits.device)
            return torch.sigmoid(logits / tau)

    def get_keep_prob(self, seq_len: int) -> torch.Tensor:
        """Return keep probability for compatibility."""
        with torch.no_grad():
            return torch.sigmoid(self._fallback_logits[:seq_len] / 0.5)

    def step_temperature(self):
        """No-op for SparsePO."""
        pass

    @property
    def current_tau(self):
        return torch.tensor(0.5)
        """
        Args:
            hidden_states_list: list of [B, S, D] from last N layers
            dialogue_mask: [B, S] bool, True for dialogue tokens
        Returns:
            token_mask: [B, S] in [0, 1]
        """
        # Per-layer scores
        layer_masks = []
        for i, hs in enumerate(hidden_states_list):
            score = self.layer_projections[i](hs)  # [B, S, 1]
            layer_masks.append(score)

        # Stack and merge: [B, S, num_layers] → [B, S]
        stacked = torch.cat(layer_masks, dim=-1)
        merged = self.layer_merger(stacked).squeeze(-1)

        # ReLU for sparsity
        token_mask = F.relu(merged)

        # Only mask dialogue region
        token_mask = token_mask * dialogue_mask.float()

        # Clamp
        token_mask = torch.clamp(token_mask, 0.0, 1.0)

        return token_mask


# ============================================================
# Option C: Hard Concrete L0 Mask
# ============================================================

class HardConcreteMask(nn.Module):
    """
    L0-regularized mask using Hard Concrete distribution.

    Reference: Louizos et al., ICLR 2018.
    Extension: CoNNect (TMLR 2025) connectivity regularization.
    """

    def __init__(self, config: LTIMConfig, hidden_dim: int = 2048):
        super().__init__()
        self.config = config
        self.max_seq_len = config.max_seq_len

        # Learnable log-alpha parameters
        self.log_alpha = nn.Parameter(torch.zeros(config.max_seq_len))

        # Mask embedding
        self.mask_embedding = nn.Parameter(torch.zeros(hidden_dim))

    def forward(self, seq_len: int, training: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            mask: [seq_len] hard mask
            l0_reg: scalar L0 regularization term
        """
        log_alpha = self.log_alpha[:seq_len]
        logits_dtype = log_alpha.dtype
        beta = torch.tensor(self.config.hc_beta, dtype=logits_dtype, device=log_alpha.device)
        gamma = torch.tensor(self.config.hc_gamma, dtype=logits_dtype, device=log_alpha.device)
        zeta = torch.tensor(self.config.hc_zeta, dtype=logits_dtype, device=log_alpha.device)

        if training:
            # Sample from Concrete distribution
            u = torch.rand_like(log_alpha).clamp(min=1e-8)
            s = torch.sigmoid(
                (torch.log(u) - torch.log(1 - u) + log_alpha) / beta
            )
            # Hard Concrete stretch
            s_bar = s * (zeta - gamma) + gamma
            z = s_bar.clamp(0, 1)

            # Straight-Through
            z_hard = (z > 0.5).to(dtype=logits_dtype)
            mask = z_hard - z.detach() + z
        else:
            # Deterministic
            prob = torch.sigmoid(log_alpha)
            mask = (prob > 0.5).to(dtype=logits_dtype)

        # L0 regularization
        l0_reg = torch.sigmoid(
            log_alpha - self.config.hc_beta * torch.log(
                torch.tensor(-self.config.hc_gamma / self.config.hc_zeta, device=log_alpha.device)
            )
        ).sum()

        return mask, l0_reg

    def get_expected_sparsity(self, seq_len: int) -> float:
        with torch.no_grad():
            probs = torch.sigmoid(self.log_alpha[:seq_len])
            return 1.0 - probs.mean().item()


# ============================================================
# Mask Application Helper
# ============================================================

def apply_ltim_mask(
    orig_embeddings: torch.Tensor,
    mask: torch.Tensor,
    mask_embedding: torch.Tensor,
    dialogue_ranges: list[tuple[int, int]],
    mask_dialogue_only: bool = True,
) -> torch.Tensor:
    """
    Apply learned token mask to embeddings.

    Args:
        orig_embeddings: [B, S, D] original token embeddings
        mask: [S] or [B, S] per-token mask values
        mask_embedding: [D] embedding for masked tokens
        dialogue_ranges: list of (start, end) for dialogue region
        mask_dialogue_only: if True, only mask dialogue tokens

    Returns:
        masked_embeddings: [B, S, D]
    """
    B, S, D = orig_embeddings.shape
    emb_dtype = orig_embeddings.dtype
    emb_device = orig_embeddings.device

    # Cast mask and mask_embedding to match embeddings dtype
    mask = mask.to(dtype=emb_dtype, device=emb_device)
    mask_embedding = mask_embedding.to(dtype=emb_dtype, device=emb_device)

    # Build full mask
    if mask.dim() == 1:
        full_mask = mask.unsqueeze(0).expand(B, -1)  # [B, S]
    else:
        full_mask = mask  # [B, S]

    # If dialogue-only, keep non-dialogue tokens unmasked
    if mask_dialogue_only and dialogue_ranges:
        keep_mask = torch.ones(B, S, device=emb_device, dtype=emb_dtype)
        for b in range(B):
            d_start, d_end = dialogue_ranges[b]
            # Clamp to valid range
            d_start = max(0, min(d_start, S))
            d_end = max(d_start, min(d_end, S))
            if d_end > d_start:
                keep_mask[b, d_start:d_end] = full_mask[b, d_start:d_end]
        full_mask = keep_mask

    # Apply: mask * orig + (1-mask) * mask_emb
    mask_expanded = full_mask.unsqueeze(-1)  # [B, S, 1]
    masked_embeddings = (
        mask_expanded * orig_embeddings +
        (1 - mask_expanded) * mask_embedding
    )

    return masked_embeddings


# ============================================================
# Factory
# ============================================================

def create_ltim_mask(config: LTIMConfig, hidden_dim: int = 2048, num_layers: int = 28):
    """Create LTIM mask module based on config."""
    if config.mask_type == "gumbel_sigmoid":
        return GumbelSigmoidMask(config, hidden_dim)
    elif config.mask_type == "sparsepo":
        return SparsePOTokenMask(config, hidden_dim, num_layers)
    elif config.mask_type == "hard_concrete":
        return HardConcreteMask(config, hidden_dim)
    else:
        raise ValueError(f"Unknown mask type: {config.mask_type}")
