"""Stage III co-attention fusion (paper Fig. 2 right block).

Ported from merits-l-text/src/models/stage3_fusion.py — encoder-agnostic,
only cfg.model.text_dim differs (RoBERTa Stage II 1024 -> Llama Stage II 2048).
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class Stage3Fusion(nn.Module):
    def __init__(
        self,
        text_dim: int,
        audio_dim: int,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_labels: int = 4,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})."
            )
        self.proj_text = nn.Linear(text_dim, hidden_dim)
        self.proj_audio = nn.Linear(audio_dim, hidden_dim)

        self.text_attends_audio = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        self.audio_attends_text = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        self.ln_text = nn.LayerNorm(hidden_dim)
        self.ln_audio = nn.LayerNorm(hidden_dim)

        self.fuse_fc = nn.Linear(4 * hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_labels)

        self.num_labels = num_labels
        self.hidden_dim = hidden_dim

    def forward(
        self,
        text_features: torch.Tensor,
        audio_features: torch.Tensor,
        mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ):
        t = self.proj_text(text_features)
        a = self.proj_audio(audio_features)

        key_padding_mask = ~mask
        t_attn, _ = self.text_attends_audio(
            t, a, a, key_padding_mask=key_padding_mask, need_weights=False,
        )
        a_attn, _ = self.audio_attends_text(
            a, t, t, key_padding_mask=key_padding_mask, need_weights=False,
        )

        t_out = self.ln_text(t + self.dropout(t_attn))
        a_out = self.ln_audio(a + self.dropout(a_attn))

        fused = torch.cat([t_out, a_out, t, a], dim=-1)
        h = F.relu(self.fuse_fc(fused))
        h = self.dropout(h)
        logits = self.classifier(h)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=-100,
            )
        return {"loss": loss, "logits": logits, "fused_hidden": h}


def build_stage3_fusion(cfg) -> Stage3Fusion:
    return Stage3Fusion(
        text_dim=int(cfg.text_dim),
        audio_dim=int(cfg.audio_dim),
        hidden_dim=int(cfg.hidden_dim),
        num_heads=int(cfg.num_heads),
        num_labels=int(cfg.num_labels),
        dropout=float(cfg.dropout),
    )
