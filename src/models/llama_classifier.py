"""Llama-3.1 + LoRA + linear classification head for 4-class SER.

Analogous to the RoBERTa-FT + FC head used in the MERITS-L baseline
(merits-l-text). Key differences from RoBERTa:
  - Llama is decoder-only → we pool over `last_hidden_state` (mean by default)
  - LoRA-adapts only q_proj / v_proj (add k_proj / o_proj via config if needed)
  - Base model kept in fp16 to fit LoRA training under 25 GB VRAM
"""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import AutoModel


class LlamaClassifier(nn.Module):
    """Llama base + LoRA + mean-pool + linear head."""

    def __init__(
        self,
        model_id: str = "meta-llama/Meta-Llama-3.1-8B",
        num_labels: int = 4,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        target_modules: Optional[List[str]] = None,
        pool: str = "mean",
        head_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if pool not in ("mean", "last"):
            raise ValueError(f"pool must be 'mean' or 'last', got {pool!r}")
        self.pool = pool
        self.num_labels = num_labels

        self.base = AutoModel.from_pretrained(model_id, torch_dtype=torch.float16)
        hidden = self.base.config.hidden_size  # 4096 for Llama-3.1-8B

        lora = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=target_modules or ["q_proj", "v_proj"],
            lora_dropout=lora_dropout,
            bias="none",
            task_type="FEATURE_EXTRACTION",
        )
        self.base = get_peft_model(self.base, lora)

        # Classification head — kept in fp32 for numerical stability.
        self.dropout = nn.Dropout(head_dropout)
        self.classifier = nn.Linear(hidden, num_labels)

    def _pool(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        hidden_states: (B, T, H)
        attention_mask: (B, T)  1 = real, 0 = pad
        returns: (B, H)
        """
        if self.pool == "mean":
            mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
            summed = (hidden_states * mask).sum(dim=1)
            denom = mask.sum(dim=1).clamp(min=1e-6)
            return summed / denom
        # "last": take the last real token per sample.
        lengths = attention_mask.sum(dim=1) - 1                     # (B,)
        idx = lengths.clamp(min=0).view(-1, 1, 1).expand(-1, 1, hidden_states.size(-1))
        return hidden_states.gather(1, idx).squeeze(1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        return_features: bool = False,
    ):
        outputs = self.base(input_ids=input_ids, attention_mask=attention_mask)
        h = outputs.last_hidden_state                                # (B, T, H)
        pooled = self._pool(h, attention_mask)                       # (B, H) fp16
        pooled_fp32 = pooled.float()                                 # head in fp32
        logits = self.classifier(self.dropout(pooled_fp32))          # (B, num_labels)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
        out = {"loss": loss, "logits": logits}
        if return_features:
            out["features"] = pooled_fp32
        return out

    def save_lora(self, path: str) -> None:
        """Save the LoRA adapter + classification head only (not full model)."""
        self.base.save_pretrained(path)
        torch.save(
            {"classifier": self.classifier.state_dict(),
             "num_labels": self.num_labels, "pool": self.pool},
            f"{path}/head.pt",
        )

    def load_lora(self, path: str, device: torch.device) -> None:
        """Load LoRA adapter + head. If the saved head's num_labels differs
        from the current classifier's, the head is skipped (fresh init) so
        the same LoRA checkpoint can be reused across tasks with different
        label spaces.
        """
        self.base.load_adapter(path, adapter_name="default")
        head_path = f"{path}/head.pt"
        try:
            head = torch.load(head_path, map_location=device, weights_only=True)
        except FileNotFoundError:
            print(f"[load_lora] no head.pt under {path}; leaving classifier at fresh init.")
            return
        saved_num = int(head.get("num_labels", -1))
        if saved_num == self.num_labels:
            self.classifier.load_state_dict(head["classifier"])
            print(f"[load_lora] loaded LoRA + head ({saved_num}-class) from {path}")
        else:
            print(f"[load_lora] loaded LoRA only from {path} "
                  f"(saved head is {saved_num}-class, current is {self.num_labels}-class -> fresh head)")


def build_llama_classifier(cfg) -> LlamaClassifier:
    return LlamaClassifier(
        model_id=str(cfg.model_id),
        num_labels=int(cfg.num_labels),
        lora_r=int(cfg.get("lora_r", 16)) if hasattr(cfg, "get") else 16,
        lora_alpha=int(cfg.get("lora_alpha", 32)) if hasattr(cfg, "get") else 32,
        lora_dropout=float(cfg.get("lora_dropout", 0.1)) if hasattr(cfg, "get") else 0.1,
        target_modules=list(cfg.get("target_modules", ["q_proj", "v_proj"])) if hasattr(cfg, "get") else ["q_proj", "v_proj"],
        pool=str(cfg.get("pool", "mean")) if hasattr(cfg, "get") else "mean",
        head_dropout=float(cfg.get("head_dropout", 0.1)) if hasattr(cfg, "get") else 0.1,
    )
