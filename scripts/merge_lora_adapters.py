"""Merge two LoRA adapters in weight space: tau_a + lambda * tau_b.

A LoRA adapter is a task vector — Delta_W = (alpha/r) * B @ A is exactly
theta_ft - theta_base in low-rank form — so task arithmetic applies to it
directly.

The experiment this exists for: fine-tuning Llama on MSP-PODCAST and *then* on
IEMOCAP did not help (0.8550 +/- 0.0072 against 0.8567 +/- 0.0139 without the
pre-training step). Sequential fine-tuning overwrites; merging interpolates. If
forgetting is why the LLM-supervised pre-training failed to transfer, merging
should recover it. If merging does not help either, the MSP signal genuinely
carries nothing for an 8B encoder, independent of how it is integrated — which
strengthens the scale-dependence result rather than weakening it.

**Adding the A and B matrices is wrong.** (B1+B2)(A1+A2) != B1A1 + B2A2. This
concatenates instead, which is exact:

    A_cat = [A_a ; lambda * A_b]   (2r, in)
    B_cat = [B_a , B_b]            (out, 2r)
    B_cat @ A_cat = B_a @ A_a + lambda * B_b @ A_b

Rank doubles, so alpha doubles too, keeping peft's scaling = alpha / r
unchanged. The result is a plain adapter directory that any
`--lora-ckpt` argument accepts.

Usage:
    python -m scripts.merge_lora_adapters \
        --adapter-a outputs/iemocap_text_llama_stage1/best \
        --adapter-b outputs/msppodcast_llama_pretrain/best \
        --lam 0.3 \
        --out outputs/merged/iemocap_plus_msp_lam03

    # lambda = 0 reproduces adapter A exactly — run it once as a pipeline control.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, Tuple

import torch

# peft names the two factors like this inside the checkpoint keys.
A_TAG, B_TAG = "lora_A", "lora_B"


def _pair_key(key: str) -> Tuple[str, str]:
    """Split a checkpoint key into (module path, 'lora_A' | 'lora_B')."""
    if A_TAG in key:
        return key.replace(A_TAG, "{}"), A_TAG
    if B_TAG in key:
        return key.replace(B_TAG, "{}"), B_TAG
    return key, ""


def merge_lora_tensors(
    sd_a: Dict[str, torch.Tensor],
    sd_b: Dict[str, torch.Tensor],
    lam: float,
) -> Dict[str, torch.Tensor]:
    """Concatenate two LoRA state dicts so the deltas add exactly.

    Both must cover the same modules with the same rank; anything else means
    the two adapters do not live in the same subspace and must not be merged.
    """
    if set(sd_a) != set(sd_b):
        only_a = sorted(set(sd_a) - set(sd_b))[:3]
        only_b = sorted(set(sd_b) - set(sd_a))[:3]
        raise ValueError(
            f"adapters cover different modules; a-only e.g. {only_a}, b-only e.g. {only_b}"
        )

    merged: Dict[str, torch.Tensor] = {}
    for key, ta in sd_a.items():
        tb = sd_b[key]
        _, tag = _pair_key(key)
        if tag == A_TAG:
            # (r, in) each -> (2r, in); lambda rides on B's rows.
            if ta.shape[1] != tb.shape[1]:
                raise ValueError(f"{key}: input dims differ, {ta.shape} vs {tb.shape}")
            merged[key] = torch.cat([ta.float(), lam * tb.float()], dim=0).to(ta.dtype)
        elif tag == B_TAG:
            # (out, r) each -> (out, 2r)
            if ta.shape[0] != tb.shape[0]:
                raise ValueError(f"{key}: output dims differ, {ta.shape} vs {tb.shape}")
            merged[key] = torch.cat([ta.float(), tb.float()], dim=1).to(ta.dtype)
        else:
            # Anything that is not a LoRA factor (rare: biases, embeddings) has
            # no concatenation semantics, so it is a weighted sum instead.
            merged[key] = (ta.float() + lam * tb.float()).to(ta.dtype)
    return merged


def verify(sd_a, sd_b, merged, lam: float, atol: float = 1e-4) -> int:
    """Check B_cat @ A_cat == B_a @ A_a + lambda * B_b @ A_b on every module."""
    checked = 0
    for key in sd_a:
        stem, tag = _pair_key(key)
        if tag != A_TAG:
            continue
        b_key = stem.format(B_TAG)
        if b_key not in sd_a:
            continue
        want = (sd_a[b_key].float() @ sd_a[key].float()
                + lam * sd_b[b_key].float() @ sd_b[key].float())
        got = merged[b_key].float() @ merged[key].float()
        err = (got - want).abs().max().item()
        if err > atol:
            raise AssertionError(f"{stem}: merged delta differs by {err:.2e}")
        checked += 1
    return checked


def load_adapter(path: Path) -> Tuple[Dict[str, torch.Tensor], dict]:
    cfg_path = path / "adapter_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"{cfg_path} not found — is {path} a peft adapter directory?")
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    st = path / "adapter_model.safetensors"
    if st.exists():
        from safetensors.torch import load_file
        return load_file(str(st)), cfg
    bin_path = path / "adapter_model.bin"
    if bin_path.exists():
        return torch.load(str(bin_path), map_location="cpu", weights_only=True), cfg
    raise FileNotFoundError(f"no adapter_model.safetensors or .bin under {path}")


def check_compatible(cfg_a: dict, cfg_b: dict) -> None:
    """Two task vectors are only addable if they span the same subspace."""
    for field in ("r", "lora_alpha", "base_model_name_or_path"):
        if cfg_a.get(field) != cfg_b.get(field):
            raise ValueError(f"adapters disagree on {field}: "
                             f"{cfg_a.get(field)} vs {cfg_b.get(field)}")
    if sorted(cfg_a.get("target_modules") or []) != sorted(cfg_b.get("target_modules") or []):
        raise ValueError(f"different target_modules: {cfg_a.get('target_modules')} "
                         f"vs {cfg_b.get('target_modules')}")
    for flag in ("use_rslora", "use_dora"):
        if cfg_a.get(flag) or cfg_b.get(flag):
            raise ValueError(f"{flag} changes how the delta is computed; "
                             f"the concatenation identity here assumes plain LoRA")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter-a", required=True, help="kept at weight 1.0")
    ap.add_argument("--adapter-b", required=True, help="scaled by --lam")
    ap.add_argument("--lam", required=True, type=float,
                    help="weight on adapter B; 0 reproduces A exactly, "
                         "negative subtracts the task direction")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sd_a, cfg_a = load_adapter(Path(args.adapter_a))
    sd_b, cfg_b = load_adapter(Path(args.adapter_b))
    check_compatible(cfg_a, cfg_b)

    merged = merge_lora_tensors(sd_a, sd_b, args.lam)
    n = verify(sd_a, sd_b, merged, args.lam)
    print(f"merged {len(merged)} tensors; delta identity verified on {n} modules")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg_out = dict(cfg_a)
    # Rank doubled by concatenation, so alpha doubles to hold scaling = alpha/r.
    cfg_out["r"] = int(cfg_a["r"]) * 2
    cfg_out["lora_alpha"] = int(cfg_a["lora_alpha"]) * 2
    cfg_out["_merged_from"] = {"a": str(args.adapter_a), "b": str(args.adapter_b),
                               "lam": args.lam}
    (out / "adapter_config.json").write_text(json.dumps(cfg_out, indent=2), encoding="utf-8")

    from safetensors.torch import save_file
    save_file(merged, str(out / "adapter_model.safetensors"))

    # The head belongs to whichever task A was trained for; feature extraction
    # never calls it, but carrying it keeps the directory loadable as-is.
    head = Path(args.adapter_a) / "head.pt"
    if head.exists():
        shutil.copy2(head, out / "head.pt")

    print(f"wrote {out}  (r {cfg_a['r']} -> {cfg_out['r']}, "
          f"alpha {cfg_a['lora_alpha']} -> {cfg_out['lora_alpha']}, lam={args.lam})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
