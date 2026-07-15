"""Stage III trainer: co-attention fusion of text + audio Stage II hiddens.

Uses precomputed dialogue-context utt-level hiddens from Stage II:
    data/cache/iemocap_text_llama_stage2_utt.pt   (from Llama Stage II)
    data/cache/iemocap_audio_stage2_utt.pt        (reused from merits-l-text)

Usage:
    python -m src.train_stage3 --config configs/iemocap_stage3_llama.yaml
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.optim import AdamW
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from .data.multimodal_dataset import build_multimodal_loaders
from .models.stage3_fusion import build_stage3_fusion
from .utils.config import AttrDict, load_config
from .utils.logging import RunLogger
from .utils.metrics import compute_metrics, detailed_report
from .utils.seed import set_seed


@torch.no_grad()
def evaluate(model, loader, device, label_names) -> Dict:
    model.eval()
    all_preds: List[int] = []
    all_labels: List[int] = []
    losses: List[float] = []
    for batch in loader:
        text = batch["text_features"].to(device, non_blocking=True)
        audio = batch["audio_features"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        out = model(text_features=text, audio_features=audio, mask=mask, labels=labels)
        losses.append(out["loss"].item())
        preds = out["logits"].argmax(dim=-1)
        valid = mask
        all_preds.extend(preds[valid].cpu().tolist())
        all_labels.extend(labels[valid].cpu().tolist())
    metrics = compute_metrics(all_labels, all_preds, label_names=label_names)
    metrics["loss"] = float(np.mean(losses)) if losses else float("nan")
    metrics["_preds"] = all_preds
    metrics["_labels"] = all_labels
    return metrics


def train(cfg: AttrDict) -> None:
    set_seed(int(cfg.seed), deterministic=bool(cfg.get("deterministic", False)))
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.snapshot.yaml").write_text(
        json.dumps(dict(cfg), indent=2, default=str), encoding="utf-8"
    )

    runlog = RunLogger(
        output_dir=out_dir,
        run_name=str(cfg.run_name),
        use_wandb=bool(cfg.logging.use_wandb),
        wandb_project=str(cfg.logging.wandb_project),
        wandb_config=dict(cfg),
    )
    runlog.info(f"Output dir: {out_dir.resolve()}")

    loaders = build_multimodal_loaders(
        manifest_dir=cfg.dataset.manifest_dir,
        text_features_path=cfg.dataset.text_features_path,
        audio_features_path=cfg.dataset.audio_features_path,
        batch_size=int(cfg.train.batch_size),
        eval_batch_size=int(cfg.train.eval_batch_size),
        num_workers=int(cfg.train.num_workers),
    )
    split_sizes = {k: len(v.dataset) for k, v in loaders.items()}
    runlog.info("Loader sizes (dialogues): " +
                ", ".join(f"{k}={n}" for k, n in split_sizes.items()))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_stage3_fusion(cfg.model).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    runlog.info(f"Trainable params: {n_params/1e6:.2f}M | device: {device}")

    runlog.update_wandb_config({
        "dataset/train_size": split_sizes.get("train", 0),
        "dataset/val_size":   split_sizes.get("val", 0),
        "dataset/test_size":  split_sizes.get("test", 0),
        "params/trainable":   n_params,
        "hardware/gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    })
    runlog.info("=" * 70)
    runlog.info(f"Text  features: {cfg.dataset.text_features_path}")
    runlog.info(f"Audio features: {cfg.dataset.audio_features_path}")
    runlog.info(f"Model:  text_dim={cfg.model.text_dim}  audio_dim={cfg.model.audio_dim}  "
                f"hidden={cfg.model.hidden_dim}  heads={cfg.model.num_heads}")
    runlog.info(f"Split sizes:  " + ", ".join(f"{k}={n}" for k, n in split_sizes.items()))
    runlog.info(f"Trainable:    {n_params/1e6:.2f}M")
    runlog.info(f"Batch:        {cfg.train.batch_size}")
    runlog.info(f"LR:           {cfg.train.learning_rate}  wd={cfg.train.weight_decay}")
    runlog.info(f"Epochs:       {cfg.train.epochs}  patience={cfg.train.early_stopping_patience}")
    runlog.info(f"Seed:         {cfg.seed}")
    runlog.info("=" * 70)

    no_decay = ("bias", "LayerNorm.weight")
    grouped = [
        {"params": [p for n, p in model.named_parameters()
                    if p.requires_grad and not any(nd in n for nd in no_decay)],
         "weight_decay": float(cfg.train.weight_decay)},
        {"params": [p for n, p in model.named_parameters()
                    if p.requires_grad and any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    optimizer = AdamW(grouped, lr=float(cfg.train.learning_rate))
    total_steps = len(loaders["train"]) * int(cfg.train.epochs)
    warmup_steps = int(total_steps * float(cfg.train.warmup_ratio))
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    label_names = list(cfg.dataset.label_names) if "label_names" in cfg.dataset else None

    best_score = -math.inf
    best_epoch = -1
    patience = int(cfg.train.early_stopping_patience)
    bad_epochs = 0
    log_every = int(cfg.logging.log_every)
    global_step = 0
    best_ckpt_dir = out_dir / "best"

    for epoch in range(int(cfg.train.epochs)):
        model.train()
        pbar = tqdm(loaders["train"], desc=f"epoch {epoch+1}/{cfg.train.epochs}")
        running: List[float] = []
        for batch in pbar:
            text = batch["text_features"].to(device, non_blocking=True)
            audio = batch["audio_features"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            out = model(text_features=text, audio_features=audio, mask=mask, labels=labels)
            loss = out["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.train.grad_clip))
            optimizer.step()
            scheduler.step()

            running.append(loss.item())
            global_step += 1
            if global_step % log_every == 0:
                avg = float(np.mean(running[-log_every:]))
                pbar.set_postfix(loss=f"{avg:.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")
                runlog.log_scalars(
                    {"loss": avg, "lr": scheduler.get_last_lr()[0]},
                    step=global_step, prefix="train",
                )

        val_metrics = evaluate(model, loaders["val"], device, label_names)
        runlog.log_scalars(
            {k: v for k, v in val_metrics.items() if not k.startswith("_")},
            step=global_step, prefix="val",
        )
        runlog.info(
            f"epoch {epoch+1}: val loss={val_metrics['loss']:.4f} "
            f"acc={val_metrics['accuracy']:.4f} wF1={val_metrics['weighted_f1']:.4f}"
        )

        score = val_metrics[str(cfg.train.save_best_metric)]
        if score > best_score:
            best_score = score
            best_epoch = epoch + 1
            bad_epochs = 0
            best_ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {"model_state_dict": model.state_dict(), "epoch": epoch + 1,
                 "score": score, "model_cfg": dict(cfg.model)},
                best_ckpt_dir / "stage3.pt",
            )
            runlog.info(f"  -> new best ({cfg.train.save_best_metric}={score:.4f}) saved.")
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                runlog.info(f"Early stopping at epoch {epoch+1} (no improvement {patience} epochs).")
                break

    if "test" in loaders and best_epoch > 0:
        runlog.info(f"Reloading best checkpoint from epoch {best_epoch} for test evaluation.")
        ckpt = torch.load(best_ckpt_dir / "stage3.pt", map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        test_metrics = evaluate(model, loaders["test"], device, label_names)
        runlog.log_scalars(
            {k: v for k, v in test_metrics.items() if not k.startswith("_")},
            step=global_step, prefix="test",
        )
        runlog.info(
            f"TEST | acc={test_metrics['accuracy']:.4f} "
            f"weighted_f1={test_metrics['weighted_f1']:.4f} "
            f"macro_f1={test_metrics['macro_f1']:.4f}"
        )
        report = detailed_report(test_metrics["_labels"], test_metrics["_preds"], label_names)
        (out_dir / "test_report.txt").write_text(report, encoding="utf-8")
        runlog.info("\n" + report)

    runlog.info(f"DONE. Best {cfg.train.save_best_metric} = {best_score:.4f} (epoch {best_epoch}).")
    runlog.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--override", nargs="*", default=[])
    args = parser.parse_args()

    cfg = load_config(args.config)
    for ov in args.override:
        key, _, val = ov.partition("=")
        keys = key.split(".")
        node = cfg
        for k in keys[:-1]:
            node = node[k]
        try:
            parsed = json.loads(val)
        except json.JSONDecodeError:
            parsed = val
        node[keys[-1]] = parsed

    train(cfg)


if __name__ == "__main__":
    main()
