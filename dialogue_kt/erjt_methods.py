"""
ERJT: Evidence-Rationale Joint Training for Dialogue-KT.

Implements four training modes from the ERJT v2 design:
  A1: Multi-Granularity Extractor + Plausibility Loss
  A2: UNIREX Faithfulness Loss (comprehensiveness + sufficiency)
  A3: Faith-DARE Disentanglement (R/U separation + MI minimization)
  A4: Pseudo-Label Bootstrapping (iterative self-training)
  B1/B2: Combined modes (Extractor + Faithfulness + Disentanglement)

Key simplifications for 8GB VRAM + LoRA:
  - Shared encoder + independent head modules (not separate model)
  - Frobenius norm MI approximation instead of MINE network
  - n_envs=2 for invariance loss
  - Attention mask control for comprehensiveness
  - Only last-layer hidden states used

References:
  - UNIREX (Chan et al., ICML 2022)
  - Faith-DARE (Yue et al., TPAMI 2025)
  - PLMR (Yuan et al., KDD 2025)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ============================================================
# A1: Multi-Granularity Extractor
# ============================================================

class MultiGranularityExtractor(nn.Module):
    """
    Multi-granularity evidence extractor.

    Takes hidden states from the shared Qwen encoder and outputs:
      - token_scores: [B, S] per-token evidence importance
      - turn_scores: [B, T] per-turn aggregated evidence scores

    Design follows UNIREX shared encoder + extractor head architecture.
    """

    def __init__(self, hidden_dim: int = 2048, dropout: float = 0.1):
        super().__init__()
        inner_dim = hidden_dim // 4

        self.token_scorer = nn.Sequential(
            nn.Linear(hidden_dim, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, 1),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        turn_boundaries: Optional[list] = None,
    ) -> dict:
        """
        Args:
            hidden_states: [B, S, D] last-layer hidden states
            turn_boundaries: optional list of lists, each inner list is
                [(t_start, t_end), ...] for each batch item

        Returns:
            dict with 'token_scores', optionally 'turn_scores'
        """
        B, S, D = hidden_states.shape

        # Token-level scores: [B, S]
        token_scores = self.token_scorer(hidden_states).squeeze(-1)
        token_scores = torch.sigmoid(token_scores)

        result = {'token_scores': token_scores}

        # Aggregate to turn-level if boundaries provided
        if turn_boundaries is not None:
            turn_scores_list = []
            for b in range(B):
                boundaries = turn_boundaries[b] if b < len(turn_boundaries) else []
                b_turn_scores = []
                for t_start, t_end in boundaries:
                    if t_start < S and t_end <= S and t_end > t_start:
                        turn_tok_scores = token_scores[b, t_start:t_end]
                        b_turn_scores.append(turn_tok_scores.mean())
                    else:
                        b_turn_scores.append(torch.tensor(0.0, device=hidden_states.device))
                if b_turn_scores:
                    turn_scores_list.append(torch.stack(b_turn_scores))
                else:
                    turn_scores_list.append(torch.zeros(0, device=hidden_states.device))

            # Pad to max turns
            max_turns = max(ts.shape[0] for ts in turn_scores_list) if turn_scores_list else 0
            if max_turns > 0:
                turn_scores = torch.zeros(B, max_turns, device=hidden_states.device)
                for b, ts in enumerate(turn_scores_list):
                    if ts.shape[0] > 0:
                        turn_scores[b, :ts.shape[0]] = ts
                result['turn_scores'] = turn_scores

        return result


# ============================================================
# A3: Disentanglement Module (Faith-DARE inspired)
# ============================================================

class DisentanglementModule(nn.Module):
    """
    Faith-DARE inspired disentanglement: separate hidden states into
    Rationale (R) and Non-Rationale (U) representations.

    Uses a simplified Frobenius-norm-based MI proxy instead of a full
    MINE network to reduce parameter count and memory.

    Args:
        hidden_dim: Qwen3-1.7B hidden dimension (2048)
        latent_dim: dimension of R and U projections (256)
    """

    def __init__(self, hidden_dim: int = 2048, latent_dim: int = 256):
        super().__init__()

        # R encoder: hidden → rational representation
        self.r_encoder = nn.Sequential(
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
        )

        # U encoder: hidden → non-rational representation
        self.u_encoder = nn.Sequential(
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
        )

        # Decoder: reconstruct original hidden from R+U (auxiliary)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, hidden_dim),
        )

    def forward(self, hidden_states: torch.Tensor) -> dict:
        """
        Args:
            hidden_states: [B, S, D]

        Returns:
            {
                'R': [B, S, latent_dim]
                'U': [B, S, latent_dim]
                'mi_estimate': scalar Frobenius-norm-based MI proxy
                'recon_loss': scalar reconstruction loss
            }
        """
        R = self.r_encoder(hidden_states)
        U = self.u_encoder(hidden_states)

        # Frobenius-norm-based MI proxy: ||R^T U||_F / (||R||_F * ||U||_F)
        # This measures the alignment between R and U subspaces
        # Lower value → less redundancy → lower MI
        R_flat = R.reshape(-1, R.shape[-1])  # [B*S, L]
        U_flat = U.reshape(-1, U.shape[-1])  # [B*S, L]

        cross = torch.norm(torch.mm(R_flat.T, U_flat), p='fro')
        norm_r = torch.norm(R_flat, p='fro')
        norm_u = torch.norm(U_flat, p='fro')

        mi_estimate = cross / (norm_r * norm_u + 1e-8)

        # Reconstruction loss: how well can we reconstruct original from R+U?
        combined = torch.cat([R, U], dim=-1)
        recon = self.decoder(combined)
        recon_loss = F.mse_loss(recon, hidden_states)

        return {
            'R': R,
            'U': U,
            'mi_estimate': mi_estimate,
            'recon_loss': recon_loss,
        }


# ============================================================
# Pseudo-Label Construction (for A1 Plausibility Loss)
# ============================================================

def find_kc_keyword_positions(
    input_ids: torch.Tensor,
    tokenizer,
    kc_texts: list[str],
) -> torch.Tensor:
    """
    Find token positions matching KC keywords in the input.

    Returns:
        pseudo_labels: [S] float tensor, 1.0 for KC-matching tokens, 0.0 otherwise
    """
    S = input_ids.shape[0]
    pseudo_labels = torch.zeros(S, device=input_ids.device)

    for kc_text in kc_texts:
        # Tokenize the KC text
        kc_ids = tokenizer(kc_text, add_special_tokens=False).input_ids
        if not kc_ids:
            continue

        # Search for exact subsequence matches
        ids_list = input_ids.tolist()
        for i in range(len(ids_list) - len(kc_ids) + 1):
            if ids_list[i:i + len(kc_ids)] == kc_ids:
                # Mark these positions
                pseudo_labels[i:i + len(kc_ids)] = 1.0

        # Also do partial match: single key tokens
        for kid in kc_ids:
            matches = (input_ids == kid)
            pseudo_labels[matches] = torch.maximum(
                pseudo_labels[matches],
                torch.tensor(0.5, device=input_ids.device)
            )

    return pseudo_labels


def construct_pseudo_labels_batch(
    batch: dict,
    tokenizer,
    threshold: float = 0.35,
) -> Optional[torch.Tensor]:
    """
    Construct token-level pseudo evidence labels for a batch.

    Uses KC keywords to identify evidence tokens. Marks tokens that match
    any KC keyword as potential evidence.

    Returns:
        pseudo_labels: [B, S_max] float tensor, or None if no KCs available
    """
    B = batch["input_ids"].shape[0]
    S_max = batch["input_ids"].shape[1]
    pseudo_labels = torch.zeros(B, S_max, device=batch["input_ids"].device)

    has_any_kc = False
    for b in range(B):
        meta = batch["meta_data"][b]
        kcs = meta.get("kcs", [])
        if not kcs:
            continue
        has_any_kc = True

        pl = find_kc_keyword_positions(
            batch["input_ids"][b], tokenizer, kcs
        )
        # Apply threshold to binarize
        pl = (pl > threshold).float()
        pseudo_labels[b, :pl.shape[0]] = pl

    if not has_any_kc:
        return None

    return pseudo_labels


# ============================================================
# A2: Faithfulness Loss Helpers
# ============================================================

def compute_token_importance_from_hidden(
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    """
    Compute token importance scores from hidden states.

    Uses L2 norm of hidden states as a simple importance proxy.
    Tokens with larger norm tend to carry more information.

    Args:
        hidden_states: [B, S, D]

    Returns:
        importance: [B, S] normalized importance scores
    """
    importance = torch.norm(hidden_states, p=2, dim=-1)  # [B, S]
    # Normalize per sample to [0, 1]
    imp_min = importance.min(dim=-1, keepdim=True).values
    imp_max = importance.max(dim=-1, keepdim=True).values
    importance = (importance - imp_min) / (imp_max - imp_min + 1e-8)
    return importance


def create_removal_attention_mask(
    batch_attention_mask: torch.Tensor,
    token_scores: torch.Tensor,
    k_ratio: float = 0.3,
) -> torch.Tensor:
    """
    Create an attention mask that blocks attention to the top-k% important tokens.

    In the 3D packed attention mask format [B, 1, S, S] or [B, S, S],
    removing a token means setting its column (attention TO it) to min_dtype.

    Args:
        batch_attention_mask: [B, ...] existing attention mask
        token_scores: [B, S] token importance scores
        k_ratio: fraction of tokens to remove

    Returns:
        perturbed_mask: same shape as batch_attention_mask
    """
    perturbed = batch_attention_mask.clone()
    B, S = token_scores.shape

    for b in range(B):
        scores = token_scores[b]
        k = max(1, int(S * k_ratio))
        _, top_indices = torch.topk(scores, k)

        # Set attention TO these tokens to min (block)
        if perturbed.dim() == 4:  # [B, 1, S, S]
            perturbed[b, :, :, top_indices] = torch.finfo(perturbed.dtype).min
        elif perturbed.dim() == 3:  # [B, S, S]
            perturbed[b, :, top_indices] = torch.finfo(perturbed.dtype).min

    return perturbed


# ============================================================
# Loss Functions
# ============================================================

def plausibility_loss(
    token_scores: torch.Tensor,
    pseudo_labels: torch.Tensor,
    attention_mask: torch.Tensor = None,
) -> torch.Tensor:
    """
    Token-level BCE: extractor scores should match pseudo evidence labels.

    Args:
        token_scores: [B, S] predicted evidence scores
        pseudo_labels: [B, S] pseudo evidence labels (0/1)
        attention_mask: [B, S] optional valid token mask

    Returns:
        scalar loss
    """
    # Cast pseudo_labels and attention_mask to match token_scores dtype
    pseudo_labels = pseudo_labels.to(dtype=token_scores.dtype)
    if attention_mask is not None:
        attention_mask = attention_mask.to(dtype=token_scores.dtype)
        valid = attention_mask > 0.5
        if valid.sum() == 0:
            return torch.tensor(0.0, device=token_scores.device, dtype=token_scores.dtype)
        loss = F.binary_cross_entropy(
            token_scores[valid],
            pseudo_labels[valid],
            reduction='mean',
        )
    else:
        loss = F.binary_cross_entropy(
            token_scores.reshape(-1),
            pseudo_labels.reshape(-1),
            reduction='mean',
        )
    return loss


def comprehensiveness_loss(
    full_pred: torch.Tensor,
    perturbed_pred: torch.Tensor,
    labels: torch.Tensor,
    margin: float = 0.1,
) -> torch.Tensor:
    """
    UNIREX comprehensiveness loss.

    Removing important tokens should DECREASE prediction accuracy.
    CE(full) should be lower than CE(perturbed).

    L_comp = max(-m, BCE(full) - BCE(perturbed)) + m
    """
    # Cast labels to match prediction dtype
    labels = labels.to(dtype=full_pred.dtype)
    bce_full = F.binary_cross_entropy(full_pred, labels, reduction='none')
    bce_perturbed = F.binary_cross_entropy(perturbed_pred, labels, reduction='none')

    comp_diff = bce_full - bce_perturbed  # should be negative (full is better)
    loss = F.relu(-margin - comp_diff) + margin
    return loss.mean()


# ============================================================
# Invariance Loss (Faith-DARE)
# ============================================================

def invariance_loss(
    R: torch.Tensor,
    U: torch.Tensor,
    n_envs: int = 2,
) -> torch.Tensor:
    """
    Faith-DARE style invariance loss.

    Shuffle U across batch to create counterfactual environments.
    The prediction based on R should be invariant to U shuffling.

    Simplified: measure variance of R+U_shuffled representation.
    Lower variance → R dominates, U variation doesn't matter.

    Args:
        R: [B, S, L] rationale representation
        U: [B, S, L] non-rationale representation
        n_envs: number of shuffled environments

    Returns:
        scalar invariance loss
    """
    B = R.shape[0]
    if B < 2:
        return torch.tensor(0.0, device=R.device)

    env_representations = []
    for _ in range(n_envs):
        perm = torch.randperm(B, device=U.device)
        U_shuffled = U[perm]
        env_rep = (R + U_shuffled).mean(dim=1)  # [B, L]
        env_representations.append(env_rep)

    # Variance of representations across environments
    stacked = torch.stack(env_representations, dim=0)  # [E, B, L]
    variance = stacked.var(dim=0).mean()  # mean across batch and latent dim

    return variance
