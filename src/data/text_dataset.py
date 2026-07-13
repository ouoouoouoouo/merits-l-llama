"""Utterance-level text dataset for Llama fine-tune.

Reads the same CSV manifests as merits-l-text (train.csv / val.csv / test.csv)
with columns:
    utt_id, dialogue_id, text (or transcript), label (int)

Tokenises with the Llama tokenizer; collate pads to the longest sample in the
batch.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase


class TextClassificationDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 128,
        text_col: str = "text",
        label_col: str = "label",
        utt_col: str = "utt_id",
    ) -> None:
        df = pd.read_csv(manifest_path)
        # Some manifests store text under 'transcript' — fall back to that.
        if text_col not in df.columns and "transcript" in df.columns:
            text_col = "transcript"
        for col in (text_col, label_col, utt_col):
            if col not in df.columns:
                raise KeyError(f"{manifest_path}: missing column `{col}`")

        self.texts: List[str] = df[text_col].astype(str).tolist()
        self.labels: List[int] = df[label_col].astype(int).tolist()
        self.utt_ids: List[str] = df[utt_col].astype(str).tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict:
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            return_tensors=None,
        )
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "label": self.labels[idx],
            "utt_id": self.utt_ids[idx],
        }


def _make_collate(pad_token_id: int):
    def collate(batch: List[Dict]) -> Dict[str, torch.Tensor]:
        max_len = max(len(b["input_ids"]) for b in batch)
        B = len(batch)
        input_ids = torch.full((B, max_len), pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((B, max_len), dtype=torch.long)
        labels = torch.zeros(B, dtype=torch.long)
        utt_ids: List[str] = []
        for i, b in enumerate(batch):
            L = len(b["input_ids"])
            input_ids[i, :L] = torch.tensor(b["input_ids"], dtype=torch.long)
            attention_mask[i, :L] = torch.tensor(b["attention_mask"], dtype=torch.long)
            labels[i] = b["label"]
            utt_ids.append(b["utt_id"])
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "utt_ids": utt_ids,
        }
    return collate


def build_text_loaders(
    manifest_dir: str | Path,
    model_id: str,
    batch_size: int,
    eval_batch_size: int,
    max_length: int = 128,
    num_workers: int = 2,
    tokenizer: Optional[PreTrainedTokenizerBase] = None,
):
    manifest_dir = Path(manifest_dir)
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    collate = _make_collate(pad_token_id=tokenizer.pad_token_id)

    loaders = {}
    for split in ("train", "val", "test"):
        p = manifest_dir / f"{split}.csv"
        if not p.exists():
            continue
        ds = TextClassificationDataset(p, tokenizer, max_length=max_length)
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size if split == "train" else eval_batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate,
            drop_last=False,
        )
    if "train" not in loaders:
        raise FileNotFoundError(f"No train.csv under {manifest_dir}")
    return loaders, tokenizer
