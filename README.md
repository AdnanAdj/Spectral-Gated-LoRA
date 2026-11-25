# SG-LoRA: Spectral-Gated Low-Rank Adaptation for Medical Vision Foundation Models

Complete PyTorch implementation for reproducing all experiments, baselines, ablations, corruption robustness benchmarks, statistical significance tests, and spectral interpretability figures for **SG-LoRA**.

## Quick Start (Google Colab / Single GPU)
```bash
pip install -r requirements.txt
python -m sg_lora.test_sanity
python -m sg_lora.train --dataset dermamnist --method sglora_dynamic --rank 8 --num_bands 4 --eval_corruptions
```

## Full Benchmark Suite (~4–6 GPU Hours on T4 / ~1.5 Hours on RTX 3090)
```bash
bash run_all_experiments.sh
```
Outputs are logged to `./outputs/` as per-run JSONs plus aggregated CSV/LaTeX tables (`table1_macro_f1.tex`) and layer-wise spectral gain heatmaps (`fig3_learned_spectral_gains.png`).
