"""Dialogue-level dataset for Stage II / III training.

Ported from merits-l-text unchanged — the dataset is encoder-agnostic (just
reads whatever tensors are in the features .pt file, keyed by utt_id).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


class DialogueDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        features: Dict[str, torch.Tensor],
        dialogue_col: str = "dialogue_id",
        label_col: str = "label",
        utt_col: str = "utt_id",
    ) -> None:
        df = pd.read_csv(manifest_path)
        for col in (dialogue_col, label_col, utt_col):
            if col not in df.columns:
                raise KeyError(f"{manifest_path}: missing column `{col}`")

        self.features = features
        self.dialogues: List[Dict] = []
        n_dropped = 0
        for did, group in df.groupby(dialogue_col, sort=False):
            utt_ids = group[utt_col].astype(str).tolist()
            labels = group[label_col].astype(int).tolist()
            valid = [(u, l) for u, l in zip(utt_ids, labels) if u in features]
            if not valid:
                continue
            if len(valid) != len(utt_ids):
                n_dropped += len(utt_ids) - len(valid)
            self.dialogues.append({
                "dialogue_id": str(did),
                "utt_ids": [v[0] for v in valid],
                "labels": [v[1] for v in valid],
            })
        if n_dropped:
            print(f"[DialogueDataset] {n_dropped} utts had no feature and were dropped.")

    def __len__(self) -> int:
        return len(self.dialogues)

    def __getitem__(self, idx: int) -> Dict:
        d = self.dialogues[idx]
        feats = torch.stack([self.features[u] for u in d["utt_ids"]])
        labels = torch.tensor(d["labels"], dtype=torch.long)
        return {
            "dialogue_id": d["dialogue_id"],
            "utt_ids": d["utt_ids"],
            "features": feats,
            "labels": labels,
        }


def dialogue_collate(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    max_k = max(item["features"].size(0) for item in batch)
    D = batch[0]["features"].size(1)
    B = len(batch)
    features = torch.zeros(B, max_k, D, dtype=torch.float32)
    labels = torch.full((B, max_k), -100, dtype=torch.long)
    mask = torch.zeros(B, max_k, dtype=torch.bool)
    utt_ids: List[List[str]] = []
    dialogue_ids: List[str] = []
    for i, item in enumerate(batch):
        k = item["features"].size(0)
        features[i, :k] = item["features"]
        labels[i, :k] = item["labels"]
        mask[i, :k] = True
        utt_ids.append(item["utt_ids"])
        dialogue_ids.append(item["dialogue_id"])
    return {
        "features": features,
        "labels": labels,
        "mask": mask,
        "utt_ids": utt_ids,
        "dialogue_ids": dialogue_ids,
    }


def build_dialogue_loaders(
    manifest_dir: str | Path,
    features_path: str | Path,
    batch_size: int,
    eval_batch_size: int,
    num_workers: int = 0,
):
    manifest_dir = Path(manifest_dir)
    features = torch.load(str(features_path), map_location="cpu", weights_only=True)
    print(f"Loaded {len(features)} cached features from {features_path}")

    loaders = {}
    for split in ("train", "val", "test"):
        csv_path = manifest_dir / f"{split}.csv"
        if not csv_path.exists():
            continue
        ds = DialogueDataset(csv_path, features)
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size if split == "train" else eval_batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=dialogue_collate,
            drop_last=False,
        )
    if "train" not in loaders:
        raise FileNotFoundError(f"No train.csv under {manifest_dir}")
    return loaders
