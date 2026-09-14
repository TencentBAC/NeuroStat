"""Auxiliary losses that enforce complementary representations between the
statistical (TF) branch and the semantic (TB) branch.

- :class:`SupConLoss`: supervised contrastive loss on the fused representation
  (Khosla et al., NeurIPS 2020).
- :func:`orthogonal_penalty`: per-sample squared cosine similarity between the
  TF and TB embeddings, minimised so that the two branches capture
  complementary (rather than redundant) information.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    """Supervised Contrastive Loss (Khosla et al., NeurIPS 2020).

    Operates on L2-normalised embeddings.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [B, D]  L2-normalised
            labels:   [B]     integer class labels
        """
        device = features.device
        batch_size = features.size(0)
        if batch_size <= 1:
            return torch.tensor(0.0, device=device, requires_grad=True)

        labels = labels.view(-1, 1)
        mask_pos = torch.eq(labels, labels.T).float().to(device)   # [B, B]

        sim = torch.matmul(features, features.T) / self.temperature  # [B, B]

        eye = torch.eye(batch_size, device=device)
        sim = sim - eye * 1e9
        mask_pos = mask_pos - eye  # remove diagonal

        log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
        pos_count = mask_pos.sum(dim=1).clamp(min=1)
        mean_log_prob_pos = (mask_pos * log_prob).sum(dim=1) / pos_count

        return -mean_log_prob_pos.mean()


def orthogonal_penalty(z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
    """Encourage two representation branches to capture different information.

    Minimises the **per-sample** squared cosine similarity between the two
    feature vectors (batch-averaged). When ``z_a`` and ``z_b`` have different
    last-dimension sizes, the longer one is truncated so that cosine similarity
    is well-defined.
    """
    min_dim = min(z_a.size(-1), z_b.size(-1))
    z_a_proj = F.normalize(z_a[..., :min_dim], dim=-1)
    z_b_proj = F.normalize(z_b[..., :min_dim], dim=-1)
    cos_sim = (z_a_proj * z_b_proj).sum(dim=-1)
    return (cos_sim ** 2).mean()
