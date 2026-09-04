#!/usr/bin/env bash
# One point of the merged-adapter lambda sweep: tau_iemocap + lambda * tau_msp.
#
# Runs the whole chain for a single lambda — merge, extract Stage I features,
# train Stage II, extract Stage II hiddens, then Stage III over 5 seeds. One
# lambda per GPU:
#
#   i=0; for lam in 0.1 0.3 0.5 1.0; do
#     CUDA_VISIBLE_DEVICES=$i nohup bash scripts/run_lam_sweep.sh $lam > lam$lam.log 2>&1 &
#     i=$((i+1))
#   done
#
#   python -m scripts.summarize_lam_sweep      # once they finish
#
# Every step is skipped when its output already exists, so an interrupted run
# resumes where it stopped. Delete the artefact you want rebuilt.
set -euo pipefail

LAM="${1:?usage: run_lam_sweep.sh <lambda>   e.g. 0.3}"
SEEDS="${SEEDS:-1 2 3 4 5}"
ADAPTER_A="${ADAPTER_A:-outputs/iemocap_text_llama_stage1/best}"
ADAPTER_B="${ADAPTER_B:-outputs/msppodcast_llama_pretrain/best}"

MERGED="outputs/merged/lam${LAM}"
FEATS="data/cache/iemocap_llama_features_lam${LAM}.pt"
STAGE2_DIR="outputs/stage2_lam${LAM}"
UTT="data/cache/text_stage2_utt_lam${LAM}.pt"

echo "== lambda=${LAM}  A=${ADAPTER_A}  B=${ADAPTER_B}"

# 1. merge the two task vectors (seconds, CPU)
if [ -f "${MERGED}/adapter_model.safetensors" ]; then
    echo "-- merge: ${MERGED} exists, skipping"
else
    python -m scripts.merge_lora_adapters \
        --adapter-a "${ADAPTER_A}" --adapter-b "${ADAPTER_B}" \
        --lam "${LAM}" --out "${MERGED}"
fi

# 2. Stage I features from the merged adapter (the slow step, ~20 min)
if [ -f "${FEATS}" ]; then
    echo "-- features: ${FEATS} exists, skipping"
else
    python -m scripts.extract_text_features_llama \
        --config configs/iemocap_text_llama_stage1_merged.yaml \
        --lora-ckpt "${MERGED}" \
        --out-pt "${FEATS}"
fi

# 3. Stage II
if [ -f "${STAGE2_DIR}/best/stage2.pt" ]; then
    echo "-- stage2: ${STAGE2_DIR}/best/stage2.pt exists, skipping"
else
    python -m src.train_stage2 --config configs/iemocap_text_llama_stage2.yaml \
        --override dataset.features_path="${FEATS}" \
                   output_dir="${STAGE2_DIR}" \
                   run_name="stage2_lam${LAM}"
fi

# 4. Stage II utterance hiddens. features_path is a config field, not a flag,
#    so it has to be overridden or this silently reads the default features.
if [ -f "${UTT}" ]; then
    echo "-- stage2 hiddens: ${UTT} exists, skipping"
else
    python -m scripts.extract_stage2_utt_hidden \
        --config configs/iemocap_text_llama_stage2.yaml \
        --stage2-ckpt "${STAGE2_DIR}/best/stage2.pt" \
        --override dataset.features_path="${FEATS}" \
        --out-pt "${UTT}"
fi

# 5. Stage III, 5 seeds — matches how the 0.8567 baseline was produced
#    (one Stage II run, seeds varied at Stage III).
for s in ${SEEDS}; do
    out="outputs/stage3_lam${LAM}_seed${s}"
    if [ -f "${out}/test_report.txt" ]; then
        echo "-- stage3 seed ${s}: done, skipping"
        continue
    fi
    python -m src.train_stage3 --config configs/iemocap_stage3_llama.yaml \
        --override seed="${s}" \
                   dataset.text_features_path="${UTT}" \
                   output_dir="${out}" \
                   run_name="stage3_lam${LAM}_seed${s}"
done

echo "== lambda=${LAM} done"
