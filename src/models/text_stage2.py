"""Stage II text model: Bi-GRU + multi-head self-attention over utterance features.

Ported verbatim from merits-l-text/src/models/text_stage2.py — the model
architecture is agnostic to the source encoder, only the `input_dim` in the
config changes (RoBERTa 1024 -> Llama 4096).
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class TextStage2(nn.Module):
    def __init__(
        self,
        input_dim: int = 4096,
        gru_hidden: int = 1024,
        gru_layers: int = 2,
        num_heads: int = 8,
        num_labels: int = 4,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.bigru = nn.GRU(
            input_size=input_dim,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if gru_layers > 1 else 0.0,
        )
        feature_dim = 2 * gru_hidden
        if feature_dim % num_heads != 0:
            raise ValueError(
                f"feature_dim ({feature_dim}) must be divisible by num_heads ({num_heads})."
            )
        self.attn = nn.MultiheadAttention(
            embed_dim=feature_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        self.ln_gru = nn.LayerNorm(feature_dim)
        self.ln_attn = nn.LayerNorm(feature_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(feature_dim, num_labels)
        self.num_labels = num_labels
        self.feature_dim = feature_dim

    def encode(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        gru_out, _ = self.bigru(features)
        gru_out = self.ln_gru(gru_out)
        key_padding_mask = ~mask
        attn_out, _ = self.attn(
            gru_out, gru_out, gru_out,
            key_padding_mask=key_padding_mask, need_weights=False,
        )
        x = self.ln_attn(gru_out + self.dropout(attn_out))
        return x

    def forward(
        self,
        features: torch.Tensor,
        mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ):
        x = self.encode(features, mask)
        logits = self.classifier(x)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=-100,
            )
        return {"loss": loss, "logits": logits, "features": x}


def build_text_stage2(cfg) -> TextStage2:
    return TextStage2(
        input_dim=int(cfg.input_dim),
        gru_hidden=int(cfg.gru_hidden),
        gru_layers=int(cfg.get("gru_layers", 2)) if hasattr(cfg, "get") else 2,
        num_heads=int(cfg.num_heads),
        num_labels=int(cfg.num_labels),
        dropout=float(cfg.dropout),
    )
