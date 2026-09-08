#!/usr/bin/env bash
# One point of the Stage I KL-alignment sweep: CE + lambda_kl * KL(text, audio).
#
# Runs the whole chain for a single lambda_kl — Stage I with the KL term,
# extract Stage I features, train Stage II, extract Stage II hiddens, then
# Stage III over 5 seeds. One lambda per GPU:
#
#   i=0; for lam in 0 0.1 0.5 1.0; do
#     CUDA_VISIBLE_DEVICES=$i nohup bash scripts/run_kl_sweep.sh $lam > kl$lam.log 2>&1 &
#     i=$((i+1))
#   done
#
#   python -m scripts.summarize_kl_sweep       # once they finish
#
# lambda_kl = 0 is the control: it disables the KL term entirely, so that row
# must land back on the staged Llama + CARE baseline (0.8567). If it does not,
# the chain drifted and no other row can be trusted.
#
# Note this mirrors run_lam_sweep.sh: ONE Stage I/II run per lambda, seeds
# varied at Stage III only. That is how the 0.8567 baseline was produced, so
# the numbers stay comparable. It does mean the std here does not cover Stage I
# seed noise.
#
# This lambda is NOT the merged-adapter lambda of run_lam_sweep.sh. Different
# knob, different artefact names (kl<LAM> vs lam<LAM>) — do not mix the two.
#
# Every step is skipped when its output already exists, so an interrupted run
# resumes where it stopped. Delete the artefact you want rebuilt.
set -euo pipefail

LAM="${1:?usage: run_kl_sweep.sh <lambda_kl>   e.g. 0.5   (0 = control)}"
SEEDS="${SEEDS:-1 2 3 4 5}"
STAGE1_CFG="${STAGE1_CFG:-configs/iemocap_text_llama_stage1_kl.yaml}"

STAGE1_DIR="outputs/stage1_kl${LAM}"
FEATS="data/cache/iemocap_llama_features_kl${LAM}.pt"
STAGE2_DIR="outputs/stage2_kl${LAM}"
UTT="data/cache/text_stage2_utt_kl${LAM}.pt"

echo "== lambda_kl=${LAM}  config=${STAGE1_CFG}  gpu=${CUDA_VISIBLE_DEVICES:-all}"

# 1. Stage I with the KL alignment term (~8 min on a 4090)
if [ -f "${STAGE1_DIR}/best/head.pt" ]; then
    echo "-- stage1: ${STAGE1_DIR}/best exists, skipping"
else
    python -m src.train_kl --config "${STAGE1_CFG}" \
        --override train.lambda_kl="${LAM}" \
                   output_dir="${STAGE1_DIR}" \
                   run_name="stage1_kl${LAM}"
fi

# 2. Stage I features from the KL-trained adapter (the slow step, ~20 min)
if [ -f "${FEATS}" ]; then
    echo "-- features: ${FEATS} exists, skipping"
else
    python -m scripts.extract_text_features_llama \
        --config "${STAGE1_CFG}" \
        --lora-ckpt "${STAGE1_DIR}/best" \
        --out-pt "${FEATS}"
fi

# 3. Stage II
if [ -f "${STAGE2_DIR}/best/stage2.pt" ]; then
    echo "-- stage2: ${STAGE2_DIR}/best/stage2.pt exists, skipping"
else
    python -m src.train_stage2 --config configs/iemocap_text_llama_stage2.yaml \
        --override dataset.features_path="${FEATS}" \
                   output_dir="${STAGE2_DIR}" \
                   run_name="stage2_kl${LAM}"
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

# 5. Stage III, 5 seeds
for s in ${SEEDS}; do
    out="outputs/stage3_kl${LAM}_seed${s}"
    if [ -f "${out}/test_report.txt" ]; then
        echo "-- stage3 seed ${s}: done, skipping"
        continue
    fi
    python -m src.train_stage3 --config configs/iemocap_stage3_llama.yaml \
        --override seed="${s}" \
                   dataset.text_features_path="${UTT}" \
                   output_dir="${out}" \
                   run_name="stage3_kl${LAM}_seed${s}"
done

echo "== lambda_kl=${LAM} done"
