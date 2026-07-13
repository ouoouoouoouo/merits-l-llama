# merits-l-llama

Investigating LLM-scale text/audio encoders (Llama-3.1-8B + WavLM-Large CARE)
for multimodal speech emotion recognition on IEMOCAP.

An extension of the MERITS-L pipeline (Dutta & Ganapathy, ICASSP 2025) that
upgrades the text and audio encoders to modern large-scale pretrained models:

- **Text**: RoBERTa-large (355M) → **Llama-3.1-8B** with LoRA fine-tuning
- **Audio**: WavLM-Base + CARE → **WavLM-Large + CARE-Large** (retrained from scratch)
- **Semantic alignment target** during CARE pretraining is updated from
  RoBERTa mean-pool to base Llama-3.1-8B mean-pool

The pipeline follows MERITS-L's three-stage architecture on IEMOCAP 4-class:
1. **Stage I** — per-utterance classifier on frozen encoder features
2. **Stage II** — dialogue-context Bi-GRU + self-attention
3. **Stage III** — text/audio co-attention fusion

The goal is to test whether recent LLM advances can push multimodal SER
beyond the paper's reported 0.8648 weighted F1.

Baseline reproduction (RoBERTa + WavLM-Base CARE) lives in the sibling
repository [merits-l-text](https://github.com/ouoouoouoouo/merits-l-text).
See its README for the reproduction results (final Stage III wF1 = 0.8305 ± 0.0138).

---

## Repository layout

```
merits-l-llama/
├── configs/                # per-stage YAML configs (Llama + CARE-Large variants)
├── src/
│   ├── data/               # dataset loaders (utterance + dialogue-level + multimodal)
│   ├── models/             # llama_classifier.py, text_stage2.py, stage3_fusion.py
│   ├── utils/              # config, logging, metrics
│   ├── train.py            # Stage I trainer (LoRA fine-tune Llama)
│   ├── train_stage2.py     # Stage II trainer (Bi-GRU + attn)
│   └── train_stage3.py     # Stage III trainer (co-attention fusion)
├── scripts/
│   ├── extract_llama_mean_pool.py    # Semantic target for CARE (MSP-PODCAST)
│   ├── extract_text_features_llama.py# IEMOCAP Llama features for Stage II
│   └── extract_stage2_utt_hidden.py  # utt-level hidden for Stage III
├── thesis/                 # LaTeX tables comparing baseline vs upgraded pipeline
└── requirements.txt
```

Companion audio-encoder repo (CARE-Large training):
[care-training-large](https://github.com/ouoouoouoouo/care-training-large)
*(to be created)*

---

## Status

- [ ] Phase 0 — env setup, Llama access, smoke tests
- [ ] Phase 1 — Text pipeline (Llama-3.1-8B LoRA)
- [ ] Phase 2 — Audio pipeline (WavLM-Large CARE + Llama semantic target)
- [ ] Phase 3 — Stage III co-attention fusion
- [ ] Phase 4 — Evaluation + thesis writing

## Hardware

- 2 × NVIDIA RTX 4090 (24 GB each)
- Llama-3.1-8B: LoRA fine-tune fits on one 4090 (~22 GB VRAM)
- CARE-Large: WavLM-large + Llama-4096 target fits on one 4090 (~17 GB VRAM)
- Parallelizable across the two GPUs (Text on GPU 0, Audio on GPU 1)
