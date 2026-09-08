"""Text Stage I trainer with cross-modal KL alignment.

Same as `src/train.py` (Llama-3.1-8B + LoRA + classification head), plus a
GEmo-CLAP-style KL contrastive loss that pulls the Llama utterance embedding
towards the pre-extracted CARE audio embedding of the same utterance:

    L = CE(logits, y) + lambda_kl * KL_contrastive(text_emb, audio_emb, y)

`train.lambda_kl: 0` reduces this to plain `src/train.py` behaviour, so the
same script can produce the ablation's no-KL row.

Usage:
    python -m src.train_kl --config configs/iemocap_text_llama_stage1_kl.yaml
    python -m src.train_kl --config <cfg> --override train.lambda_kl=0.5
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

from .data.text_dataset import build_text_loaders
from .models.kl_loss import KLContrastiveLoss
from .models.llama_classifier import build_llama_classifier
from .utils.config import AttrDict, load_config
from .utils.logging import RunLogger
from .utils.metrics import compute_metrics, detailed_report
from .utils.seed import set_seed


@torch.no_grad()
def evaluate(model, loader, device, label_names) -> Dict:
    """Validation / test use the CE objective only — the KL term is a training
    regulariser and would not be comparable against the no-KL runs."""
    model.eval()
    all_preds: List[int] = []
    all_labels: List[int] = []
    losses: List[float] = []
    for batch in loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        losses.append(out["loss"].item())
        preds = out["logits"].argmax(dim=-1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
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

    kl_weight = float(cfg.train.get("lambda_kl", 0.0))
    use_kl = kl_weight > 0.0
    audio_emb_path = cfg.dataset.get("audio_emb_path", None)
    if use_kl and not audio_emb_path:
        raise ValueError(
            "train.lambda_kl > 0 requires dataset.audio_emb_path (a .pt holding "
            "{utt_id: audio embedding}, e.g. data/cache/iemocap_audio_stage2_utt.pt)."
        )

    # Data — audio embeddings are only loaded when the KL term is active.
    loaders, tokenizer = build_text_loaders(
        manifest_dir=cfg.dataset.manifest_dir,
        model_id=str(cfg.model.model_id),
        batch_size=int(cfg.train.batch_size),
        eval_batch_size=int(cfg.train.eval_batch_size),
        max_length=int(cfg.train.max_length),
        num_workers=int(cfg.train.num_workers),
        audio_emb_path=audio_emb_path if use_kl else None,
        drop_last_train=use_kl,
    )
    split_sizes = {k: len(v.dataset) for k, v in loaders.items()}
    runlog.info("Loader sizes: " + ", ".join(f"{k}={n}" for k, n in split_sizes.items()))

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_llama_classifier(cfg.model).to(device)

    resume_path = cfg.model.get("resume_from", None) if hasattr(cfg.model, "get") else None
    if resume_path:
        runlog.info(f"Resuming LoRA + classifier head from {resume_path}")
        model.load_lora(resume_path, device)

    # KL contrastive loss (owns the two projections + two temperatures).
    kl_criterion = None
    if use_kl:
        train_ds = loaders["train"].dataset
        audio_dim = int(train_ds.audio_dim)
        text_dim = int(model.classifier.in_features)
        kl_criterion = KLContrastiveLoss(
            text_dim=text_dim,
            audio_dim=audio_dim,
            proj_dim=int(cfg.train.get("kl_proj_dim", 256)),
            proj_dropout=float(cfg.train.get("kl_proj_dropout", 0.0)),
        ).to(device)
        coverage = 100.0 * train_ds.n_with_audio / max(1, len(train_ds))
        runlog.info(
            f"KL contrastive loss ON: lambda={kl_weight}  "
            f"text {text_dim}-d -> {kl_criterion.proj_dim}-d <- audio {audio_dim}-d  "
            f"(train audio coverage {coverage:.1f}%)"
        )
        if train_ds.n_with_audio == 0:
            raise ValueError(
                f"No train utt_id matched an audio embedding in {audio_emb_path}; "
                "check that the manifests and the audio .pt use the same utt_id format."
            )
    else:
        runlog.info("KL contrastive loss OFF (set train.lambda_kl > 0 to enable).")

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if use_kl:
        n_train += sum(p.numel() for p in kl_criterion.parameters())
    n_all = sum(p.numel() for p in model.parameters())
    runlog.info(f"Trainable params: {n_train/1e6:.2f}M / {n_all/1e9:.2f}B  "
                f"({100*n_train/n_all:.2f}%)  device: {device}")

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    effective_bs = int(cfg.train.batch_size) * max(1, int(cfg.train.grad_accum))
    runlog.update_wandb_config({
        "dataset/train_size": split_sizes.get("train", 0),
        "dataset/val_size":   split_sizes.get("val", 0),
        "dataset/test_size":  split_sizes.get("test", 0),
        "params/trainable":   n_train,
        "params/total":       n_all,
        "params/trainable_pct": 100.0 * n_train / n_all,
        "params/effective_batch": effective_bs,
        "hardware/gpu": gpu_name,
        "hardware/gpu_count": torch.cuda.device_count(),
        "kl/lambda": kl_weight,
        "kl/proj_dim": int(cfg.train.get("kl_proj_dim", 256)) if use_kl else 0,
    })

    runlog.info("=" * 70)
    runlog.info(f"Model:        {cfg.model.model_id}")
    runlog.info(f"LoRA:         r={cfg.model.get('lora_r', 16)}  "
                f"alpha={cfg.model.get('lora_alpha', 32)}  "
                f"dropout={cfg.model.get('lora_dropout', 0.1)}  "
                f"targets={list(cfg.model.get('target_modules', ['q_proj','v_proj']))}")
    runlog.info(f"Pool:         {cfg.model.get('pool', 'mean')}")
    runlog.info(f"Dataset:      {cfg.dataset.manifest_dir}  labels={list(cfg.dataset.label_names)}")
    runlog.info(f"Split sizes:  train={split_sizes.get('train')}  "
                f"val={split_sizes.get('val')}  test={split_sizes.get('test')}")
    runlog.info(f"Trainable:    {n_train/1e6:.2f}M / {n_all/1e9:.2f}B  ({100*n_train/n_all:.2f}%)")
    runlog.info(f"Batch:        {cfg.train.batch_size} x grad_accum {cfg.train.grad_accum}  "
                f"= effective {effective_bs}")
    runlog.info(f"LR:           {cfg.train.learning_rate}  "
                f"warmup_ratio={cfg.train.warmup_ratio}  "
                f"weight_decay={cfg.train.weight_decay}")
    runlog.info(f"Epochs:       {cfg.train.epochs}  "
                f"patience={cfg.train.early_stopping_patience}")
    runlog.info(f"Seed:         {cfg.seed}")
    runlog.info(f"KL:           lambda={kl_weight}  audio={audio_emb_path if use_kl else '-'}")
    runlog.info(f"Hardware:     {gpu_name}  (device_count={torch.cuda.device_count()})")
    runlog.info("=" * 70)

    # Optimizer + scheduler. The KL projections/temperatures form a separate
    # param group so they can take their own (usually larger) LR.
    opt_name = str(cfg.train.get("optimizer", "adamw")).lower()
    param_groups = [{"params": [p for p in model.parameters() if p.requires_grad],
                     "lr": float(cfg.train.learning_rate)}]
    if use_kl:
        kl_lr = float(cfg.train.get("kl_lr", cfg.train.learning_rate))
        param_groups.append({"params": list(kl_criterion.parameters()), "lr": kl_lr})
        runlog.info(f"KL param group LR: {kl_lr}")

    if opt_name in ("adamw8bit", "adam8bit", "8bit"):
        import bitsandbytes as bnb  # type: ignore
        optimizer = bnb.optim.AdamW8bit(
            param_groups,
            lr=float(cfg.train.learning_rate),
            weight_decay=float(cfg.train.weight_decay),
        )
        runlog.info("Optimizer: bitsandbytes AdamW8bit (paged m/v to fp8).")
    else:
        optimizer = AdamW(
            param_groups,
            lr=float(cfg.train.learning_rate),
            weight_decay=float(cfg.train.weight_decay),
        )
        runlog.info("Optimizer: torch AdamW (fp32 m/v).")
    total_steps = len(loaders["train"]) * int(cfg.train.epochs) // max(1, int(cfg.train.grad_accum))
    warmup_steps = int(total_steps * float(cfg.train.warmup_ratio))
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    label_names = list(cfg.dataset.label_names) if "label_names" in cfg.dataset else None

    best_score = -math.inf
    best_epoch = -1
    patience = int(cfg.train.early_stopping_patience)
    bad_epochs = 0
    log_every = int(cfg.logging.log_every)
    global_step = 0
    grad_accum = max(1, int(cfg.train.grad_accum))
    grad_clip = float(cfg.train.grad_clip)

    best_ckpt_dir = out_dir / "best"

    for epoch in range(int(cfg.train.epochs)):
        model.train()
        if use_kl:
            kl_criterion.train()
        pbar = tqdm(loaders["train"], desc=f"epoch {epoch+1}/{cfg.train.epochs}")
        running_ce: List[float] = []
        running_kl: List[float] = []
        optimizer.zero_grad(set_to_none=True)

        for micro_step, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                return_features=use_kl,      # out["features"] = pooled fp32 (B, H)
            )
            ce_loss = out["loss"]

            kl_loss = torch.zeros((), device=device)
            if use_kl:
                audio_emb = batch["audio_emb"].to(device, non_blocking=True)
                has_audio = batch["has_audio"].to(device, non_blocking=True)
                kl_loss = kl_criterion(out["features"], audio_emb, labels, has_audio)

            total_loss = ce_loss + kl_weight * kl_loss

            (total_loss / grad_accum).backward()
            running_ce.append(ce_loss.item())
            running_kl.append(float(kl_loss.item()))

            if (micro_step + 1) % grad_accum == 0:
                clip_params = [p for p in model.parameters() if p.requires_grad]
                if use_kl:
                    clip_params = clip_params + list(kl_criterion.parameters())
                torch.nn.utils.clip_grad_norm_(clip_params, grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % log_every == 0:
                    w = log_every * grad_accum
                    avg_ce = float(np.mean(running_ce[-w:]))
                    avg_kl = float(np.mean(running_kl[-w:]))
                    pbar.set_postfix(
                        ce=f"{avg_ce:.4f}",
                        kl=(f"{avg_kl:.4f}" if use_kl else "off"),
                        lr=f"{scheduler.get_last_lr()[0]:.2e}",
                    )
                    scalars = {
                        "loss_ce": avg_ce,
                        "loss_total": avg_ce + kl_weight * avg_kl,
                        "lr": scheduler.get_last_lr()[0],
                    }
                    if use_kl:
                        scalars["loss_kl"] = avg_kl
                        scalars["kl_temp_t"] = float(
                            kl_criterion.logit_scale_t.detach()
                            .clamp(math.log(1.0), math.log(100.0)).exp())
                        scalars["kl_temp_a"] = float(
                            kl_criterion.logit_scale_a.detach()
                            .clamp(math.log(1.0), math.log(100.0)).exp())
                    runlog.log_scalars(scalars, step=global_step, prefix="train")

        # Validation
        val_metrics = evaluate(model, loaders["val"], device, label_names)
        runlog.log_scalars(
            {k: v for k, v in val_metrics.items() if not k.startswith("_")},
            step=global_step, prefix="val",
        )
        runlog.info(
            f"epoch {epoch+1}: val loss={val_metrics['loss']:.4f} "
            f"acc={val_metrics['accuracy']:.4f} wF1={val_metrics['weighted_f1']:.4f}"
            + (f" | train kl={float(np.mean(running_kl)):.4f}" if use_kl else "")
        )

        score = val_metrics[str(cfg.train.save_best_metric)]
        if score > best_score:
            best_score = score
            best_epoch = epoch + 1
            bad_epochs = 0
            best_ckpt_dir.mkdir(parents=True, exist_ok=True)
            model.save_lora(str(best_ckpt_dir))
            if use_kl:
                # Not needed downstream (Stage II reads the pooled features, not
                # the projections), but keeps the alignment head reproducible.
                torch.save(kl_criterion.state_dict(), best_ckpt_dir / "kl_criterion.pt")
            with (best_ckpt_dir / "best_metadata.json").open("w") as f:
                json.dump({"epoch": epoch + 1, "score": score,
                           "metric": str(cfg.train.save_best_metric),
                           "lambda_kl": kl_weight}, f, indent=2)
            runlog.info(f"  -> new best ({cfg.train.save_best_metric}={score:.4f}) saved.")
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                runlog.info(f"Early stopping at epoch {epoch+1} (no improvement {patience} epochs).")
                break

    if "test" in loaders and best_epoch > 0:
        runlog.info(f"Reloading best LoRA adapter from epoch {best_epoch} for test evaluation.")
        model.load_lora(str(best_ckpt_dir), device)
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
