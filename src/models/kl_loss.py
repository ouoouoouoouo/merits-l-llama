"""KL-based cross-modal contrastive loss (GEmo-CLAP style) for Stage I.

Aligns the Llama utterance embedding (4096-d, before the classifier FC) with a
pre-extracted CARE audio embedding (e.g. the 256-d WavLM+CARE Stage II vector
reused from merits-l-text), so that Stage I text features already carry some
acoustic emotion structure before Stage II / Stage III fusion.

The two modalities live in different spaces (4096 vs 256), so each side gets a
small linear projection into a shared `proj_dim` space before the cosine
similarity matrix is formed. Those projections + the two learned temperatures
are the only parameters this module owns; they are trained jointly with the
LoRA adapters and are saved alongside the checkpoint (kl_criterion.pt).
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class KLContrastiveLoss(nn.Module):
    """Audio-text KL contrastive loss (GEmo-CLAP Eq. 4).

    Minimises KL( softmax(M_e) || log_softmax(C) ) in both directions, where
    C is the temperature-scaled cross-modal cosine similarity matrix and
    M_e[i, j] = 1 iff emotion(i) == emotion(j). Unlike vanilla CLAP (identity
    target), this treats every same-emotion pair in the batch as a positive.

    Args:
        text_dim:  dim of the text utterance embedding (Llama-3.1-8B -> 4096).
        audio_dim: dim of the audio embedding (WavLM+CARE Stage II -> 256).
        proj_dim:  shared projection dim.
        proj_dropout: dropout applied before each projection.
    """

    def __init__(
        self,
        text_dim: int = 4096,
        audio_dim: int = 256,
        proj_dim: int = 256,
        proj_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.text_dim = int(text_dim)
        self.audio_dim = int(audio_dim)
        self.proj_dim = int(proj_dim)

        self.dropout = nn.Dropout(proj_dropout)
        # Bias-free projections keep the mapping purely directional, which is
        # what cosine similarity cares about.
        self.text_proj = nn.Linear(self.text_dim, self.proj_dim, bias=False)
        self.audio_proj = nn.Linear(self.audio_dim, self.proj_dim, bias=False)

        # Learned temperatures, initialised at log(1/0.07) ~= 2.66 (CLIP/CLAP).
        self.logit_scale_t = nn.Parameter(torch.ones([]) * math.log(1.0 / 0.07))
        self.logit_scale_a = nn.Parameter(torch.ones([]) * math.log(1.0 / 0.07))

    def encode(self, text_emb: torch.Tensor, audio_emb: torch.Tensor):
        """Project + L2-normalise both modalities into the shared space."""
        t = self.text_proj(self.dropout(text_emb.float()))
        a = self.audio_proj(self.dropout(audio_emb.float()))
        return F.normalize(t, dim=-1), F.normalize(a, dim=-1)

    def forward(
        self,
        text_emb: torch.Tensor,
        audio_emb: torch.Tensor,
        labels: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            text_emb:   (B, text_dim)  — pooled Llama embedding BEFORE the head.
            audio_emb:  (B, audio_dim) — pre-extracted CARE audio embedding.
            labels:     (B,)           — integer emotion label (0..C-1).
            valid_mask: (B,) bool      — False for utterances with no audio
                                         embedding; those rows are dropped.

        Returns:
            Scalar loss. Returns 0 (no grad path) when fewer than 2 usable rows
            remain, since a contrastive loss needs at least one pair.
        """
        if valid_mask is not None:
            valid_mask = valid_mask.to(text_emb.device).bool()
            if valid_mask.sum() < 2:
                return text_emb.sum() * 0.0
            text_emb = text_emb[valid_mask]
            audio_emb = audio_emb[valid_mask]
            labels = labels[valid_mask]
        elif text_emb.size(0) < 2:
            return text_emb.sum() * 0.0

        t, a = self.encode(text_emb, audio_emb)

        # Emotion ground-truth matrix: M_e[i, j] = 1 iff labels[i] == labels[j].
        M_e = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()

        # Clamp temperatures to [1, 100] as in CLIP, to keep logits bounded.
        scale_t = self.logit_scale_t.clamp(math.log(1.0), math.log(100.0)).exp()
        scale_a = self.logit_scale_a.clamp(math.log(1.0), math.log(100.0)).exp()

        C_t = scale_t * (t @ a.T)   # text queries, audio keys
        C_a = scale_a * (a @ t.T)   # audio queries, text keys

        S_Me = F.softmax(M_e, dim=-1)   # soft emotion-aware target (GEmo-CLAP)

        kl_t = F.kl_div(F.log_softmax(C_t, dim=-1), S_Me, reduction="batchmean")
        kl_a = F.kl_div(F.log_softmax(C_a, dim=-1), S_Me, reduction="batchmean")
        return 0.5 * (kl_t + kl_a)
