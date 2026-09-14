"""NeuroStat: dual-branch (TF + TB) detector for machine-generated text.

High-level forward pipeline::

    text ──► CausalLM ──► logits  ──► TF branch (log-prob 1D-CNN
                       │               + Macro-State Residual
                       │                 Modulation via global
                       │                 uncertainty indicators)
                       └► hidden  ──► TB branch (attention-pooled semantics)

    Fusion:  [z_tf ; z_tb] ──► FusionClassifier ──► 2-class logits

The training objective is::

    L = L_CE + lambda_supcon * L_SupCon + lambda_orth * L_Orth
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from .features import (
    TFFeatureExtractor,
    TBFeatureExtractor,
    compute_tf_features,
)
from .losses import SupConLoss, orthogonal_penalty
from .utils import resolve_model_path


# ---------------------------------------------------------------------------
# Classifier head
# ---------------------------------------------------------------------------

class FusionClassifier(nn.Module):
    """MLP head that maps the fused representation to class logits."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_labels: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_labels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class NeuroStatDetector(nn.Module):
    """TF + TB dual-branch detector with Macro-State Residual Modulation.

    Architecture:
        * A shared causal-LM backbone emits logits + last-layer hidden states
          in a single forward pass.
        * The TF branch runs a 1D-CNN over the per-token log-probability
          curve, then residually modulates its features using two global
          uncertainty indicators (mean entropy, mean log-rank).
        * The TB branch attention-pools the last-layer hidden states.
        * The two branch embeddings are concatenated and mapped to class
          logits by :class:`FusionClassifier`.

    Objective: cross-entropy + a supervised contrastive term on the fused
    representation + an orthogonal penalty encouraging the two branches to
    carry complementary information.
    """

    def __init__(
        self,
        backbone: nn.Module,
        tokenizer: AutoTokenizer,
        hidden_size: int,
        tf_dim: int = 64,
        tb_dim: int = 64,
        cls_hidden_dim: int = 128,
        cls_dropout: float = 0.1,
        num_labels: int = 2,
        supcon_temperature: float = 0.07,
        lambda_supcon: float = 0.1,
        lambda_orth: float = 1e-3,
        max_length: int = 512,
    ):
        super().__init__()
        self.backbone = backbone
        self.tokenizer = tokenizer
        self.hidden_size = hidden_size
        self.max_length = max_length
        self.backbone_frozen = not any(p.requires_grad for p in backbone.parameters())

        self.lambda_supcon = lambda_supcon
        self.lambda_orth = lambda_orth

        # TF branch: 1D-CNN + Macro-State Residual Modulation
        self.tf_extractor = TFFeatureExtractor(
            out_dim=tf_dim,
            max_length=max_length,
        )
        # TB branch: attention-pooled semantics
        self.tb_extractor = TBFeatureExtractor(
            hidden_size=hidden_size, out_dim=tb_dim,
        )

        fuse_dim = tf_dim + tb_dim
        self.classifier = FusionClassifier(
            input_dim=fuse_dim,
            hidden_dim=cls_hidden_dim,
            num_labels=num_labels,
            dropout=cls_dropout,
        )

        self.supcon_proj = nn.Linear(fuse_dim, 128)
        self.supcon_loss_fn = SupConLoss(temperature=supcon_temperature)

        # dummy buffer for head dtype probing under DataParallel
        self.register_buffer("_dtype_probe", torch.tensor(0.0), persistent=False)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _backbone_forward(self, input_ids, attention_mask):
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        return outputs.logits, outputs.hidden_states[-1]

    def _extract_features(self, lm_logits, last_hidden, input_ids, attention_mask):
        head_dtype = self._dtype_probe.dtype
        last_hidden = last_hidden.to(head_dtype)
        lm_logits = lm_logits.to(head_dtype)

        token_lp, mask, side_stats = compute_tf_features(
            lm_logits, input_ids, attention_mask,
        )
        if self.backbone_frozen:
            token_lp = token_lp.detach()
            side_stats = side_stats.detach()

        z_tf = self.tf_extractor(token_lp, mask, side_stats=side_stats)
        z_tb = self.tb_extractor(last_hidden, attention_mask)

        fused = torch.cat([z_tf, z_tb], dim=-1)
        return fused, z_tf, z_tb

    # ------------------------------------------------------------------
    # Forward / inference
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
        **kwargs,
    ):
        lm_logits, last_hidden = self._backbone_forward(input_ids, attention_mask)
        fused, z_tf, z_tb = self._extract_features(
            lm_logits, last_hidden, input_ids, attention_mask,
        )

        cls_logits = self.classifier(fused)
        result = {"logits": cls_logits, "fused": fused}

        if labels is not None:
            loss_ce = F.cross_entropy(cls_logits, labels)
            proj = F.normalize(self.supcon_proj(fused), dim=-1)
            loss_supcon = self.supcon_loss_fn(proj, labels)
            loss_orth = orthogonal_penalty(z_tf, z_tb)

            total_loss = (
                loss_ce
                + self.lambda_supcon * loss_supcon
                + self.lambda_orth * loss_orth
            )
            result.update({
                "loss": total_loss,
                "loss_ce": loss_ce,
                "loss_supcon": loss_supcon,
                "loss_orth": loss_orth,
            })

        return result

    @torch.no_grad()
    def predict_scores(self, input_texts: list[str]) -> list[float]:
        """Inference helper: return P(AIGC) for each text."""
        tokenized = self.tokenizer(
            input_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(next(self.parameters()).device)

        out = self.forward(
            input_ids=tokenized["input_ids"],
            attention_mask=tokenized["attention_mask"],
        )
        probs = torch.softmax(out["logits"], dim=-1)
        return probs[:, 1].cpu().tolist()


# ---------------------------------------------------------------------------
# Builder utilities
# ---------------------------------------------------------------------------

def load_causal_lm_backbone(
    model_name_or_path: str,
    cache_dir: str | None = None,
    freeze: bool = False,
):
    """Load a CausalLM + tokenizer. Optionally freeze every parameter."""
    resolved_path = resolve_model_path(model_name_or_path, cache_dir)
    backbone = AutoModelForCausalLM.from_pretrained(
        resolved_path,
        trust_remote_code=True,
        torch_dtype="auto",
        output_hidden_states=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(resolved_path, trust_remote_code=True)

    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
            backbone.resize_token_embeddings(len(tokenizer))

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = (
            tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
        )

    if freeze:
        for param in backbone.parameters():
            param.requires_grad = False
        backbone.eval()

    return backbone, tokenizer, resolved_path, backbone.config.hidden_size


def build_neurostat_detector(
    model_name_or_path: str,
    cache_dir: str | None = None,
    freeze_backbone: bool = False,
    **kwargs,
):
    """Convenience builder: returns ``(model, tokenizer, resolved_path)``."""
    backbone, tokenizer, resolved_path, hidden_size = load_causal_lm_backbone(
        model_name_or_path, cache_dir, freeze=freeze_backbone,
    )
    model = NeuroStatDetector(
        backbone=backbone,
        tokenizer=tokenizer,
        hidden_size=hidden_size,
        **kwargs,
    )
    return model, tokenizer, resolved_path
