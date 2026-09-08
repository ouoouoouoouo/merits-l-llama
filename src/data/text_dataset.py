"""Utterance-level text dataset for Llama fine-tune.

Reads the same CSV manifests as merits-l-text (train.csv / val.csv / test.csv)
with columns:
    utt_id, dialogue_id, text (or transcript), label (int)

Tokenises with the Llama tokenizer; collate pads to the longest sample in the
batch.

Optionally attaches a pre-extracted audio embedding per utterance (keyed by
utt_id) so Stage I can run a cross-modal KL contrastive loss against it — see
`src/models/kl_loss.py` and `src/train_kl.py`. Utterances with no matching
audio embedding get a zero vector plus `has_audio=False`, and are dropped from
the contrastive term rather than from the batch.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase


def load_audio_embeddings(path: str | Path) -> Dict[str, torch.Tensor]:
    """Load a {utt_id: 1-D float tensor} dict saved by the audio pipeline.

    Any .pt holding such a mapping works — e.g. the WavLM+CARE Stage II
    utterance hiddens (256-d) copied from merits-l-text.
    """
    obj = torch.load(str(path), map_location="cpu", weights_only=True)
    if not isinstance(obj, dict) or not obj:
        raise ValueError(f"{path}: expected a non-empty dict of utt_id -> tensor")
    emb = {str(k): torch.as_tensor(v).float().flatten() for k, v in obj.items()}
    dims = {t.numel() for t in emb.values()}
    if len(dims) != 1:
        raise ValueError(f"{path}: audio embeddings have mixed dims {sorted(dims)}")
    return emb


class TextClassificationDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 128,
        text_col: str = "text",
        label_col: str = "label",
        utt_col: str = "utt_id",
        audio_embeddings: Optional[Dict[str, torch.Tensor]] = None,
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

        self.audio_embeddings = audio_embeddings
        self.audio_dim = 0
        self.n_with_audio = 0
        if audio_embeddings:
            self.audio_dim = next(iter(audio_embeddings.values())).numel()
            self.n_with_audio = sum(1 for u in self.utt_ids if u in audio_embeddings)

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict:
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            return_tensors=None,
        )
        item = {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "label": self.labels[idx],
            "utt_id": self.utt_ids[idx],
        }
        if self.audio_embeddings is not None:
            emb = self.audio_embeddings.get(self.utt_ids[idx])
            item["audio_emb"] = emb if emb is not None else torch.zeros(self.audio_dim)
            item["has_audio"] = emb is not None
        return item


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
        out = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "utt_ids": utt_ids,
        }
        if "audio_emb" in batch[0]:
            out["audio_emb"] = torch.stack([b["audio_emb"] for b in batch]).float()
            out["has_audio"] = torch.tensor([b["has_audio"] for b in batch], dtype=torch.bool)
        return out
    return collate


def build_text_loaders(
    manifest_dir: str | Path,
    model_id: str,
    batch_size: int,
    eval_batch_size: int,
    max_length: int = 128,
    num_workers: int = 2,
    tokenizer: Optional[PreTrainedTokenizerBase] = None,
    audio_emb_path: Optional[str | Path] = None,
    drop_last_train: bool = False,
):
    manifest_dir = Path(manifest_dir)
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    collate = _make_collate(pad_token_id=tokenizer.pad_token_id)

    audio_embeddings = None
    if audio_emb_path:
        audio_embeddings = load_audio_embeddings(audio_emb_path)
        dim = next(iter(audio_embeddings.values())).numel()
        print(f"Loaded {len(audio_embeddings)} audio embeddings ({dim}-d) "
              f"from {audio_emb_path}")

    loaders = {}
    for split in ("train", "val", "test"):
        p = manifest_dir / f"{split}.csv"
        if not p.exists():
            continue
        ds = TextClassificationDataset(
            p, tokenizer, max_length=max_length, audio_embeddings=audio_embeddings
        )
        if audio_embeddings is not None:
            print(f"  {split}: {ds.n_with_audio}/{len(ds)} utterances have an audio embedding")
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size if split == "train" else eval_batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate,
            # The contrastive term is computed in-batch, so a size-1 tail batch
            # is useless for it; drop_last_train avoids that.
            drop_last=(drop_last_train and split == "train"),
        )
    if "train" not in loaders:
        raise FileNotFoundError(f"No train.csv under {manifest_dir}")
    return loaders, tokenizer
