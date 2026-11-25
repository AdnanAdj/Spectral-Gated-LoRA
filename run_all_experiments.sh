#!/usr/bin/env bash
set -e

echo "=== STEP 1: Mid-Way Unit Sanity Checks ==="
python -m sg_lora.test_sanity

echo "=== STEP 2: Main 5-Dataset x 7-Method x 3-Seed Benchmark ==="
SEEDS=(42 123 2026)
METHODS=(linear_probe lora dora fourierft sglora_static sglora_dynamic full_ft)
DATASETS=(breastmnist retinamnist dermamnist bloodmnist pathmnist)

for DS in "${DATASETS[@]}"; do
  FRAC="1.0"
  if [ "$DS" == "pathmnist" ]; then
    FRAC="0.2"  # Stratified 20% for single-GPU low-budget benchmark (set 1.0 if RTX 3090/4090)
  fi
  for M in "${METHODS[@]}"; do
    for S in "${SEEDS[@]}"; do
      CORR_FLAG=""
      if [ "$DS" == "dermamnist" ] || [ "$DS" == "breastmnist" ]; then
        CORR_FLAG="--eval_corruptions"
      fi
      python -m sg_lora.train \
        --dataset "$DS" --method "$M" --rank 8 --num_bands 4 \
        --epochs 20 --batch_size 32 --train_fraction "$FRAC" --seed "$S" $CORR_FLAG
    done
  done
done

echo "=== STEP 3: Ablation Studies (Rank r & Radial Bands K) ==="
for DS in dermamnist retinamnist breastmnist; do
  for R in 4 16; do
    for S in "${SEEDS[@]}"; do
      python -m sg_lora.train --dataset "$DS" --method sglora_dynamic --rank "$R" --num_bands 4 --seed "$S"
    done
  done
  for K in 2 8; do
    for S in "${SEEDS[@]}"; do
      python -m sg_lora.train --dataset "$DS" --method sglora_dynamic --rank 8 --num_bands "$K" --seed "$S"
    done
  done
done

echo "=== STEP 4: Aggregate Results, Significance Tests & Spectral Heatmaps ==="
python -m sg_lora.analyze_results
echo "ALL EXPERIMENTS & ANALYSIS COMPLETE!"
