"""Feature extraction modules for NeuroStat.

Two heterogeneous signals are computed from a single causal-LM backbone pass:

1. **TF branch (token-level statistical features).** Uncompressed token-level
   log-probability curves are processed by a 1D-CNN. Global uncertainty
   indicators (mean entropy and mean log-rank) drive a residual sigmoid gate
   that adaptively recalibrates the CNN feature map — the proposed
   *Macro-State Residual Modulation*.

2. **TB branch (semantic features).** Deep semantic hidden states are
   aggregated via a learnable attention pooling.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Token-level statistical feature (TF) extraction
# ---------------------------------------------------------------------------

def compute_tf_features(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
):
    """Compute token-level log-probability statistics from teacher-forcing logits.

    Args:
        logits:         [B, T, V]  full causal-LM logits
        input_ids:      [B, T]     token ids
        attention_mask: [B, T]     1 = real token, 0 = pad

    Returns:
        token_log_probs: [B, T-1]   per-token log-prob of the actual next token
        mask:            [B, T-1]   valid-token mask (shifted)
        side_stats:      [B, 2]     (mean_entropy, mean_logrank) macro-state
                                    uncertainty indicators.
    """
    # shift: logits[t] predicts input_ids[t+1]
    shift_logits = logits[:, :-1, :]               # [B, T-1, V]
    shift_labels = input_ids[:, 1:]                # [B, T-1]
    shift_mask = attention_mask[:, 1:].float()     # [B, T-1]

    log_probs = F.log_softmax(shift_logits, dim=-1)  # [B, T-1, V]
    token_log_probs = log_probs.gather(
        dim=-1, index=shift_labels.unsqueeze(-1)
    ).squeeze(-1)                                    # [B, T-1]

    lengths = shift_mask.sum(dim=-1).clamp(min=1)    # [B]
    probs = torch.softmax(shift_logits, dim=-1)      # [B, T-1, V]

    # mean entropy
    entropy = -(probs * log_probs).sum(dim=-1)                     # [B, T-1]
    mean_entropy = (entropy * shift_mask).sum(dim=-1) / lengths    # [B]

    # mean log-rank of the actual token
    actual_logits = shift_logits.gather(
        dim=-1, index=shift_labels.unsqueeze(-1)
    )                                                              # [B, T-1, 1]
    ranks = (shift_logits > actual_logits).sum(dim=-1).float() + 1
    log_ranks = torch.log(ranks)
    mean_logrank = (log_ranks * shift_mask).sum(dim=-1) / lengths  # [B]

    side_stats = torch.stack([mean_entropy, mean_logrank], dim=-1)  # [B, 2]
    return token_log_probs, shift_mask, side_stats


# ---------------------------------------------------------------------------
# TF feature extractor (1D-CNN + Macro-State Residual Modulation)
# ---------------------------------------------------------------------------

class TFFeatureExtractor(nn.Module):
    """Fixed-dim TF vector from a per-token log-prob sequence.

    Pipeline::

        token_log_probs ─► 1D-CNN ─► x  ┐
                                        ├─► x * (1 + sigmoid(W * side_stats))
        side_stats ────────► gate_net ──┘            (Macro-State Residual
                                                      Modulation)
        ───────────────────────────────► Linear ──► z_tf (out_dim)
    """

    CNN_INTERNAL_DIM = 64
    N_SIDE_STATS = 2  # mean_entropy, mean_logrank

    def __init__(
        self,
        out_dim: int = 64,
        max_length: int = 512,
    ):
        super().__init__()
        self.out_dim = out_dim

        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, self.CNN_INTERNAL_DIM, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        # Macro-State Residual Modulation: residual sigmoid gate
        self.gate_net = nn.Sequential(
            nn.Linear(self.N_SIDE_STATS, self.CNN_INTERNAL_DIM),
            nn.Sigmoid(),
        )

        self.proj = nn.Linear(self.CNN_INTERNAL_DIM, out_dim)

    def forward(
        self,
        token_log_probs: torch.Tensor,
        mask: torch.Tensor,
        side_stats: torch.Tensor,
    ) -> torch.Tensor:
        x = (token_log_probs * mask).unsqueeze(1)    # [B, 1, T-1]
        x = self.conv(x).squeeze(-1)                 # [B, 64]

        # Macro-State Residual Modulation
        g = self.gate_net(side_stats)                # [B, 64]
        x = x * (1.0 + g)

        return self.proj(x)


# ---------------------------------------------------------------------------
# Semantic (TB) feature extractor
# ---------------------------------------------------------------------------

class AttentionPooling(nn.Module):
    """Learnable attention pooling over token hidden states."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden_states: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        scores = self.attn(hidden_states).squeeze(-1)
        scores = scores.masked_fill(mask == 0, float("-inf"))
        weights = F.softmax(scores, dim=-1).unsqueeze(-1)
        pooled = (hidden_states * weights).sum(dim=1)
        return pooled


class TBFeatureExtractor(nn.Module):
    """Pool semantic hidden states into a fixed-dim TB vector via attention."""

    def __init__(self, hidden_size: int, out_dim: int = 64):
        super().__init__()
        self.pool = AttentionPooling(hidden_size)
        self.proj = nn.Linear(hidden_size, out_dim)

    def forward(self, hidden_states: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        pooled = self.pool(hidden_states, mask)
        return self.proj(pooled)
