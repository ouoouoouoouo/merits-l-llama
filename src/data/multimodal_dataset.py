"""Multimodal dialogue dataset for Stage III co-attention fusion.

Each sample = one full conversation with BOTH text and audio Stage II
utterance-level hidden vectors, aligned by utt_id.

    text_features  (utt_id -> tensor(D_t,))  — from Text  Stage II encode()
    audio_features (utt_id -> tensor(D_a,))  — from Audio Stage II encode()

Ported from merits-l-text/src/data/multimodal_dataset.py. Encoder-agnostic —
D_t / D_a differ across text encoders (RoBERTa 1024, Llama 2048).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


class MultimodalDialogueDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        text_features: Dict[str, torch.Tensor],
        audio_features: Dict[str, torch.Tensor],
        dialogue_col: str = "dialogue_id",
        label_col: str = "label",
        utt_col: str = "utt_id",
    ) -> None:
        df = pd.read_csv(manifest_path)
        for col in (dialogue_col, label_col, utt_col):
            if col not in df.columns:
                raise KeyError(f"{manifest_path}: missing column `{col}`")

        self.text_features = text_features
        self.audio_features = audio_features

        self.dialogues: List[Dict] = []
        n_dropped = 0
        for did, group in df.groupby(dialogue_col, sort=False):
            utt_ids = group[utt_col].astype(str).tolist()
            labels = group[label_col].astype(int).tolist()
            valid = [
                (u, l) for u, l in zip(utt_ids, labels)
                if u in text_features and u in audio_features
            ]
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
            print(f"[MultimodalDialogueDataset] {n_dropped} utts had missing text or audio "
                  f"features and were dropped.")

    def __len__(self) -> int:
        return len(self.dialogues)

    def __getitem__(self, idx: int) -> Dict:
        d = self.dialogues[idx]
        text = torch.stack([self.text_features[u] for u in d["utt_ids"]])
        audio = torch.stack([self.audio_features[u] for u in d["utt_ids"]])
        labels = torch.tensor(d["labels"], dtype=torch.long)
        return {
            "dialogue_id": d["dialogue_id"],
            "utt_ids": d["utt_ids"],
            "text_features": text,
            "audio_features": audio,
            "labels": labels,
        }


def multimodal_collate(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    max_k = max(item["text_features"].size(0) for item in batch)
    D_t = batch[0]["text_features"].size(1)
    D_a = batch[0]["audio_features"].size(1)
    B = len(batch)
    text = torch.zeros(B, max_k, D_t, dtype=torch.float32)
    audio = torch.zeros(B, max_k, D_a, dtype=torch.float32)
    labels = torch.full((B, max_k), -100, dtype=torch.long)
    mask = torch.zeros(B, max_k, dtype=torch.bool)
    utt_ids: List[List[str]] = []
    dialogue_ids: List[str] = []
    for i, item in enumerate(batch):
        k = item["text_features"].size(0)
        text[i, :k] = item["text_features"]
        audio[i, :k] = item["audio_features"]
        labels[i, :k] = item["labels"]
        mask[i, :k] = True
        utt_ids.append(item["utt_ids"])
        dialogue_ids.append(item["dialogue_id"])
    return {
        "text_features": text,
        "audio_features": audio,
        "labels": labels,
        "mask": mask,
        "utt_ids": utt_ids,
        "dialogue_ids": dialogue_ids,
    }


def build_multimodal_loaders(
    manifest_dir: str | Path,
    text_features_path: str | Path,
    audio_features_path: str | Path,
    batch_size: int,
    eval_batch_size: int,
    num_workers: int = 0,
):
    manifest_dir = Path(manifest_dir)
    text_features = torch.load(str(text_features_path), map_location="cpu", weights_only=True)
    audio_features = torch.load(str(audio_features_path), map_location="cpu", weights_only=True)
    print(f"Loaded {len(text_features)} text + {len(audio_features)} audio Stage II hiddens.")

    loaders = {}
    for split in ("train", "val", "test"):
        csv_path = manifest_dir / f"{split}.csv"
        if not csv_path.exists():
            continue
        ds = MultimodalDialogueDataset(csv_path, text_features, audio_features)
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size if split == "train" else eval_batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=multimodal_collate,
            drop_last=False,
        )
    if "train" not in loaders:
        raise FileNotFoundError(f"No train.csv under {manifest_dir}")
    return loaders
