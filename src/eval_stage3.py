"""Eval-only script for Stage III fusion checkpoints.

Loads a trained stage3.pt, runs test, prints weighted F1, macro F1, UAR,
per-class metrics, and confusion matrix. Should reproduce the numbers
already printed at the end of a training run bit-for-bit, given the same
checkpoint, batch size, and device.

Usage:
    python -m src.eval_stage3 \
        --config configs/iemocap_stage3_llama.yaml \
        --stage3-ckpt outputs/iemocap_stage3_llama/best/stage3.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score, accuracy_score, recall_score, classification_report, confusion_matrix

from .data.multimodal_dataset import build_multimodal_loaders
from .models.stage3_fusion import build_stage3_fusion
from .utils.config import load_config
from .utils.seed import set_seed


@torch.no_grad()
def _predict(model, loader, device):
    model.eval()
    preds, labels = [], []
    for batch in loader:
        text = batch["text_features"].to(device, non_blocking=True)
        audio = batch["audio_features"].to(device, non_blocking=True)
        m = batch["mask"].to(device, non_blocking=True)
        lab = batch["labels"].to(device, non_blocking=True)
        out = model(text_features=text, audio_features=audio, mask=m, labels=lab)
        p = out["logits"].argmax(dim=-1)
        preds.extend(p[m].cpu().tolist())
        labels.extend(lab[m].cpu().tolist())
    return np.asarray(labels), np.asarray(preds)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--stage3-ckpt", required=True, type=str)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--seed", default=None, type=int,
                        help="Only affects dataloader shuffle for train split; eval order doesn't affect metrics.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.seed is not None:
        set_seed(int(args.seed))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    loaders = build_multimodal_loaders(
        manifest_dir=cfg.dataset.manifest_dir,
        text_features_path=cfg.dataset.text_features_path,
        audio_features_path=cfg.dataset.audio_features_path,
        batch_size=int(cfg.train.batch_size),
        eval_batch_size=int(cfg.train.eval_batch_size),
        num_workers=int(cfg.train.num_workers),
    )
    if args.split not in loaders:
        raise SystemExit(f"Split {args.split!r} not available (loaders: {list(loaders)})")

    model = build_stage3_fusion(cfg.model).to(device)
    ckpt = torch.load(args.stage3_ckpt, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded: {args.stage3_ckpt}")
    print(f"  epoch = {ckpt.get('epoch')}, saved val score = {ckpt.get('score'):.4f}")

    labels, preds = _predict(model, loaders[args.split], device)

    label_names = list(cfg.dataset.label_names)
    wf1  = f1_score(labels, preds, average="weighted", zero_division=0)
    mf1  = f1_score(labels, preds, average="macro",    zero_division=0)
    acc  = accuracy_score(labels, preds)
    uar  = recall_score(labels, preds, average="macro", zero_division=0)  # macro recall = UAR
    per_class_f1 = f1_score(labels, preds, average=None,
                            labels=list(range(len(label_names))), zero_division=0)

    print()
    print(f"===== {args.split.upper()} on {len(labels)} utterances =====")
    print(f"  accuracy      = {acc:.4f}")
    print(f"  weighted F1   = {wf1:.4f}")
    print(f"  macro F1      = {mf1:.4f}")
    print(f"  UAR (macro R) = {uar:.4f}")
    print()
    for name, f in zip(label_names, per_class_f1):
        print(f"  F1[{name:8s}] = {f:.4f}")
    print()
    print(classification_report(labels, preds, target_names=label_names,
                                digits=4, zero_division=0))
    print("Confusion matrix (rows = true, cols = pred):")
    print(confusion_matrix(labels, preds))


if __name__ == "__main__":
    main()
