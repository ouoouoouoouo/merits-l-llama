"""Extract per-utterance Llama features for Stage II Bi-GRU.

Loads the best LoRA adapter from Stage I, runs inference on train/val/test
manifests, and saves a dict `{utt_id: tensor(4096,)}` to a .pt file.

Usage:
    python -m scripts.extract_text_features_llama \
        --config configs/iemocap_text_llama_stage1.yaml \
        --lora-ckpt outputs/iemocap_text_llama_stage1/best \
        --out-pt   data/cache/iemocap_llama_features.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import torch
from tqdm import tqdm

from src.data.text_dataset import build_text_loaders
from src.models.llama_classifier import build_llama_classifier
from src.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--lora-ckpt", required=True, type=str,
                        help="Directory containing the LoRA adapter (best/) from Stage I.")
    parser.add_argument("--out-pt", required=True, type=str)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    loaders, tokenizer = build_text_loaders(
        manifest_dir=cfg.dataset.manifest_dir,
        model_id=str(cfg.model.model_id),
        batch_size=1,
        eval_batch_size=1,
        max_length=int(cfg.train.max_length),
        num_workers=0,
    )
    print(f"Split sizes: " + ", ".join(f"{k}={len(v.dataset)}" for k, v in loaders.items()))

    model = build_llama_classifier(cfg.model).to(device)
    print(f"Loading LoRA adapter from {args.lora_ckpt} ...")
    model.load_lora(args.lora_ckpt, device)
    model.eval()

    hidden: Dict[str, torch.Tensor] = {}
    total = 0
    with torch.no_grad():
        for split_name in ("train", "val", "test"):
            if split_name not in loaders:
                continue
            for batch in tqdm(loaders[split_name], desc=f"extract {split_name}"):
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                utt_id = batch["utt_ids"][0]
                out = model(input_ids=input_ids, attention_mask=attention_mask, return_features=True)
                hidden[utt_id] = out["features"][0].detach().cpu().to(torch.float32)
                total += 1

    out = Path(args.out_pt)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(hidden, out)
    sample = next(iter(hidden.values()))
    print(f"Saved {out}  ({total} utts, per-utt shape {tuple(sample.shape)})")


if __name__ == "__main__":
    main()
