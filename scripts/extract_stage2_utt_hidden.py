"""Extract utterance-level Stage II hidden features for Stage III fusion.

For each split (train/val/test), for each dialogue, run the trained Stage II
model's encode() which returns (K, 2*gru_hidden) — the utterance-level
representation AFTER Bi-GRU + multi-head self-attention. Save keyed by utt_id.

Usage:
    # Text (Llama-based Stage II)
    python -m scripts.extract_stage2_utt_hidden \
        --config configs/iemocap_text_llama_stage2.yaml \
        --stage2-ckpt outputs/iemocap_text_llama_stage2/best/stage2.pt \
        --out-pt data/cache/iemocap_text_llama_stage2_utt.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import torch
from tqdm import tqdm

from src.data.dialogue_dataset import build_dialogue_loaders
from src.models.text_stage2 import build_text_stage2
from src.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str,
                        help="Same YAML used to train the Stage II checkpoint.")
    parser.add_argument("--stage2-ckpt", required=True, type=str,
                        help="Path to Stage II best/stage2.pt")
    parser.add_argument("--out-pt", required=True, type=str)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    loaders = build_dialogue_loaders(
        manifest_dir=cfg.dataset.manifest_dir,
        features_path=cfg.dataset.features_path,
        batch_size=1,
        eval_batch_size=1,
        num_workers=0,
    )

    model = build_text_stage2(cfg.model).to(device)
    ckpt = torch.load(args.stage2_ckpt, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded {args.stage2_ckpt}  (epoch {ckpt.get('epoch')}, "
          f"score {ckpt.get('score', float('nan')):.4f})")

    hidden: Dict[str, torch.Tensor] = {}
    total_utts = 0

    with torch.no_grad():
        for split_name in ("train", "val", "test"):
            if split_name not in loaders:
                continue
            for batch in tqdm(loaders[split_name], desc=f"extract {split_name}"):
                features = batch["features"].to(device, non_blocking=True)
                mask = batch["mask"].to(device, non_blocking=True)
                utt_ids = batch["utt_ids"][0]
                encoded = model.encode(features, mask)
                encoded = encoded.squeeze(0).cpu().to(torch.float32)
                for i, uid in enumerate(utt_ids):
                    hidden[uid] = encoded[i]
                    total_utts += 1

    out = Path(args.out_pt)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(hidden, out)
    sample = next(iter(hidden.values()))
    print(f"Saved {out}  ({total_utts} utts, per-utt shape {tuple(sample.shape)})")


if __name__ == "__main__":
    main()
