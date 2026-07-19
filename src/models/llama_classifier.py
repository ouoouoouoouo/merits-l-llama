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
    """Llama base + (optional LoRA) + mean-pool + linear head.

    Two modes:
      * `use_lora=True`  — freeze base, train only q/v LoRA adapters + head.
                           ~7M trainable, fits 4090 / Blackwell 5000.
      * `use_lora=False` — full fine-tune all 8B params + head.
                           ~80-100 GB VRAM with Adam, requires Blackwell 6000.
    """

    def __init__(
        self,
        model_id: str = "meta-llama/Meta-Llama-3.1-8B",
        num_labels: int = 4,
        use_lora: bool = True,
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
        self.use_lora = use_lora

        # LoRA: fp16 base (only adapters train, base frozen — fp16 is fine).
        # Full FT: bf16 base — Blackwell has native bf16, no loss-scaling needed,
        # and fp32 (~165 GB with Adam) does not fit on 102 GB Blackwell 6000.
        base_dtype = torch.float16 if use_lora else torch.bfloat16
        self.base = AutoModel.from_pretrained(model_id, torch_dtype=base_dtype)
        hidden = self.base.config.hidden_size  # 4096 for Llama-3.1-8B

        if use_lora:
            lora = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=target_modules or ["q_proj", "v_proj"],
                lora_dropout=lora_dropout,
                bias="none",
                task_type="FEATURE_EXTRACTION",
            )
            self.base = get_peft_model(self.base, lora)
        # If not use_lora, all base params train by default. Gradient checkpointing
        # was tried but not needed on Blackwell 6000 (bf16 + AdamW8bit peaks ~75%
        # of the 97 GB VRAM with batch=2, seq=128).

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
        """Save either the LoRA adapter (small) or the full base model.

        For LoRA runs, saves the adapter (~30 MB) + head. For full FT runs,
        saves the whole base (~16 GB for Llama-8B) + head.
        """
        self.base.save_pretrained(path)
        torch.save(
            {"classifier": self.classifier.state_dict(),
             "num_labels": self.num_labels, "pool": self.pool,
             "use_lora": self.use_lora},
            f"{path}/head.pt",
        )

    def load_lora(self, path: str, device: torch.device) -> None:
        """Load LoRA adapter OR full base weights + head.

        Auto-detects whether ``path`` contains a LoRA adapter (adapter_model.*)
        or a full model dump (model.safetensors / pytorch_model.bin). Head
        num_labels mismatch is handled by falling back to fresh init.
        """
        from pathlib import Path as _P
        p = _P(path)
        has_adapter = any((p / f).exists() for f in (
            "adapter_config.json", "adapter_model.safetensors", "adapter_model.bin"
        ))
        has_full = any((p / f).exists() for f in (
            "model.safetensors", "pytorch_model.bin",
            "model.safetensors.index.json", "pytorch_model.bin.index.json",
        ))

        if has_adapter:
            self.base.load_adapter(path, adapter_name="default")
            print(f"[load_lora] loaded LoRA adapter from {path}")
        elif has_full:
            # Full base checkpoint (from a use_lora=False run).
            from transformers import AutoModel
            print(f"[load_lora] loading full base weights from {path} ...")
            loaded = AutoModel.from_pretrained(path, torch_dtype=self.base.dtype)
            self.base.load_state_dict(loaded.state_dict(), strict=False)
            del loaded
            torch.cuda.empty_cache()
            print(f"[load_lora] loaded full base from {path}")
        else:
            raise FileNotFoundError(
                f"No adapter_* or model.* files found under {path}. "
                f"Directory contents: {sorted(x.name for x in p.iterdir())[:10]}"
            )

        head_path = p / "head.pt"
        if not head_path.exists():
            print(f"[load_lora] no head.pt under {path}; leaving classifier at fresh init.")
            return
        head = torch.load(str(head_path), map_location=device, weights_only=True)
        saved_num = int(head.get("num_labels", -1))
        if saved_num == self.num_labels:
            self.classifier.load_state_dict(head["classifier"])
            print(f"[load_lora] loaded head ({saved_num}-class) from {path}")
        else:
            print(f"[load_lora] head mismatch (saved {saved_num}-class, current "
                  f"{self.num_labels}-class) -> fresh head")


def build_llama_classifier(cfg) -> LlamaClassifier:
    _get = cfg.get if hasattr(cfg, "get") else lambda k, d=None: d
    return LlamaClassifier(
        model_id=str(cfg.model_id),
        num_labels=int(cfg.num_labels),
        use_lora=bool(_get("use_lora", True)),
        lora_r=int(_get("lora_r", 16)),
        lora_alpha=int(_get("lora_alpha", 32)),
        lora_dropout=float(_get("lora_dropout", 0.1)),
        target_modules=list(_get("target_modules", ["q_proj", "v_proj"])),
        pool=str(_get("pool", "mean")),
        head_dropout=float(_get("head_dropout", 0.1)),
    )
