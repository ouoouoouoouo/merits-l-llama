"""Collect a Stage I KL-alignment sweep into one table.

Reads `outputs/stage3_kl<LAM>_seed<S>/metrics.jsonl`, taking the last row with
`prefix=test` — the same convention as summarize_lam_sweep.py. Also reports the
Stage I number for each lambda (`outputs/stage1_kl<LAM>/metrics.jsonl`), so it
is visible whether a Stage I gain actually survives to Stage III.

lambda_kl = 0 is the control: it disables the KL term, so that row must land
back on the staged Llama + CARE baseline. If it does not, something in the
extract -> Stage II -> Stage III chain drifted and no other row means anything.

This is NOT the merged-adapter lambda of summarize_lam_sweep.py — that one
reads stage3_lam*, this one reads stage3_kl*.

Usage:
    python -m scripts.summarize_kl_sweep
    python -m scripts.summarize_kl_sweep --key macro_f1
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# merits-l-llama staged Llama + CARE, the pipeline this sweep is trying to beat.
BASELINE_MEAN, BASELINE_STD, BASELINE_BEST = 0.8567, 0.0139, 0.8746
_RUN_RE = re.compile(r"^stage3_kl(?P<lam>[0-9.]+)_seed(?P<seed>\d+)$")


def last_test_metric(path: Path, key: str) -> Optional[float]:
    if not path.exists():
        return None
    value = None
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("prefix") == "test" and key in row:
                value = float(row[key])
    return value


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", default="outputs", type=Path)
    ap.add_argument("--key", default="weighted_f1")
    args = ap.parse_args()

    runs: Dict[float, Dict[int, float]] = {}
    # The lambda as it was spelled on the command line ("1.0", not "1"), which
    # is what names the Stage I directory.
    lam_text: Dict[float, str] = {}
    unfinished: List[str] = []
    for d in sorted(args.outputs.glob("stage3_kl*_seed*")):
        m = _RUN_RE.match(d.name)
        if not m:
            continue
        v = last_test_metric(d / "metrics.jsonl", args.key)
        if v is None:
            unfinished.append(d.name)
            continue
        lam = float(m.group("lam"))
        lam_text[lam] = m.group("lam")
        runs.setdefault(lam, {})[int(m.group("seed"))] = v

    if not runs:
        print(f"no finished stage3_kl*_seed* runs under {args.outputs}/")
        if unfinished:
            print(f"({len(unfinished)} started but have no test metrics yet)")
        return 1

    header = (f"{'lambda_kl':>9}  {'stage1':>7}  {'n':>2}  {'fusion mean ± std':>16}  "
              f"{'best':>7}  {'vs λ=0':>7}   per-seed")
    print(f"\ntest/{args.key}, Stage I loss = CE + lambda_kl * KL(text, audio)")
    print(header)
    print("-" * (len(header) + 12))

    ctrl = runs.get(0.0)
    ctrl_mean = float(np.mean(list(ctrl.values()))) if ctrl else None

    for lam in sorted(runs):
        vals = runs[lam]
        arr = np.array([vals[s] for s in sorted(vals)])
        std = arr.std(ddof=1) if len(arr) > 1 else 0.0
        # Stage I is a single run per lambda, so it has no std to report.
        s1 = last_test_metric(
            args.outputs / f"stage1_kl{lam_text[lam]}" / "metrics.jsonl", args.key)
        s1_str = f"{s1:.4f}" if s1 is not None else "    —"
        delta = ("     —" if ctrl_mean is None or lam == 0.0
                 else f"{(arr.mean() - ctrl_mean) * 100:+6.2f}")
        per_seed = " ".join(f"{s}:{vals[s]:.4f}" for s in sorted(vals))
        print(f"{lam:>9}  {s1_str:>7}  {len(arr):>2}  {arr.mean():>8.4f} ± {std:.4f}  "
              f"{arr.max():>7.4f}  {delta:>7}   {per_seed}")

    print("-" * (len(header) + 12))
    print(f"{'staged':>9}  {'—':>7}  {5:>2}  {BASELINE_MEAN:>8.4f} ± {BASELINE_STD:.4f}  "
          f"{BASELINE_BEST:>7.4f}       —   merits-l-llama, no KL alignment")

    if ctrl_mean is None:
        print("\n*** lambda_kl = 0 has not been run. It is the pipeline control — it "
              "disables the KL term, so it must land on the staged baseline before "
              "any other row means anything. ***")
    else:
        off = abs(ctrl_mean - BASELINE_MEAN)
        verdict = ("consistent with it" if off < 2 * BASELINE_STD
                   else "*** OFF — the extract -> Stage II -> Stage III chain drifted; "
                        "fix this before reading any other row ***")
        print(f"\ncontrol: lambda_kl = 0 gives {ctrl_mean:.4f} against the staged "
              f"{BASELINE_MEAN:.4f}, {verdict}")
        print("note: seeds vary at Stage III only, so this std does not cover "
              "Stage I seed noise.")

    if unfinished:
        print(f"\nstarted but no test metrics yet: {', '.join(sorted(unfinished))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
