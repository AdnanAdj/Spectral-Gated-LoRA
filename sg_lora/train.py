"""
End-to-end training, evaluation, corruption robustness testing, and profiling script
for SG-LoRA and all PEFT baselines on MedMNIST v2 (224x224) with frozen DINOv2 ViT-B/14.
"""

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from sklearn.metrics import f1_score, balanced_accuracy_score, roc_auc_score

from sg_lora.modules import inject_peft, count_parameters, extract_spectral_profile
from sg_lora.datasets import build_medmnist_loaders


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_ece(probs: np.ndarray, targets: np.ndarray, n_bins: int = 15) -> float:
    """Expected Calibration Error (ECE) over predicted confidence bins."""
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == targets).astype(np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        in_bin = (confidences > bins[i]) & (confidences <= bins[i + 1])
        prop = in_bin.mean()
        if prop > 0:
            ece += abs(accuracies[in_bin].mean() - confidences[in_bin].mean()) * prop
    return float(ece)


@torch.no_grad()
def evaluate(model: nn.Module, loader, device: torch.device, num_classes: int) -> dict:
    model.eval()
    all_probs, all_targets = [], []
    t0 = time.perf_counter()
    n_samples = 0

    for imgs, targets in loader:
        imgs = imgs.to(device, non_blocking=True)
        targets = targets.squeeze(-1).long()
        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            logits = model(imgs)
            probs = F.softmax(logits.float(), dim=-1)
        all_probs.append(probs.cpu().numpy())
        all_targets.append(targets.numpy())
        n_samples += imgs.size(0)

    elapsed = time.perf_counter() - t0
    probs_np = np.concatenate(all_probs, axis=0)
    targets_np = np.concatenate(all_targets, axis=0)
    preds_np = probs_np.argmax(axis=1)

    macro_f1 = float(f1_score(targets_np, preds_np, average="macro", zero_division=0))
    bacc = float(balanced_accuracy_score(targets_np, preds_np))
    acc = float((preds_np == targets_np).mean())
    ece = compute_ece(probs_np, targets_np)

    try:
        if num_classes == 2:
            auroc = float(roc_auc_score(targets_np, probs_np[:, 1]))
        else:
            auroc = float(roc_auc_score(targets_np, probs_np, multi_class="ovr", average="macro"))
    except ValueError:
        auroc = float("nan")

    return {
        "macro_f1": macro_f1,
        "balanced_acc": bacc,
        "accuracy": acc,
        "auroc": auroc,
        "ece": ece,
        "latency_ms_per_img": 1000.0 * elapsed / max(n_samples, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="Train & Evaluate SG-LoRA on MedMNIST v2")
    parser.add_argument("--dataset", type=str, default="dermamnist",
                        choices=["dermamnist", "retinamnist", "bloodmnist", "breastmnist", "pathmnist"])
    parser.add_argument("--method", type=str, default="sglora_dynamic",
                        choices=["linear_probe", "full_ft", "lora", "dora", "fourierft",
                                 "sglora_static", "sglora_dynamic"])
    parser.add_argument("--backbone", type=str, default="vit_base_patch14_dinov2.lvd142m")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=16.0)
    parser.add_argument("--num_bands", type=int, default=4)
    parser.add_argument("--n_freqs", type=int, default=1000)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=None, help="Defaults: 5e-4 (PEFT), 5e-5 (Full FT)")
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--train_fraction", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_corruptions", action="store_true",
                        help="Also evaluate zero-shot robustness under Gaussian noise & blur")
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lr = args.lr if args.lr is not None else (5e-5 if args.method == "full_ft" else 5e-4)

    # 1. Load Data
    train_loader, val_loader, test_loader, num_classes, class_weights = build_medmnist_loaders(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        data_root=args.data_root,
        train_fraction=args.train_fraction,
        seed=args.seed,
    )

    # 2. Build Backbone & Inject PEFT
    model = timm.create_model(
        args.backbone, pretrained=True, num_classes=num_classes, img_size=224
    )
    model = inject_peft(
        model,
        method=args.method,
        rank=args.rank,
        alpha=args.alpha,
        dropout=args.dropout,
        num_bands=args.num_bands,
        n_freqs=args.n_freqs,
    ).to(device)

    param_stats = count_parameters(model)
    print(f"[{args.dataset} | {args.method} | r={args.rank} K={args.num_bands} | seed={args.seed}] "
          f"Trainable: {param_stats['trainable_params']:,} ({param_stats['trainable_pct']:.4f}%)")

    # 3. Optimizer, Scheduler, Loss
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=args.weight_decay)
    warmup_epochs = min(2, max(1, args.epochs // 10))

    def lr_lambda(ep):
        if ep < warmup_epochs:
            return float(ep + 1) / float(warmup_epochs)
        progress = (ep - warmup_epochs) / max(1, args.epochs - warmup_epochs)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    # 4. Training Loop with Early Stopping on Validation Macro-F1
    best_val_f1 = -1.0
    best_state = None
    no_improve = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    train_start = time.perf_counter()

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for imgs, targets in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            targets = targets.squeeze(-1).long().to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                logits = model(imgs)
                loss = criterion(logits, targets)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * imgs.size(0)

        scheduler.step()
        val_metrics = evaluate(model, val_loader, device, num_classes)
        avg_loss = running_loss / len(train_loader.dataset)
        print(f"  Epoch {epoch+1:02d}/{args.epochs} | Loss: {avg_loss:.4f} | "
              f"Val F1: {val_metrics['macro_f1']:.4f} | Val AUC: {val_metrics['auroc']:.4f}")

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                          if k in dict(model.named_parameters()) and dict(model.named_parameters())[k].requires_grad}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"  Early stopping triggered at epoch {epoch+1}.")
                break

    train_time_sec = time.perf_counter() - train_start
    peak_vram_gb = (
        torch.cuda.max_memory_allocated(device) / (1024 ** 3) if device.type == "cuda" else 0.0
    )

    # Restore best trainable weights
    if best_state is not None:
        model.load_state_dict(best_state, strict=False)

    # 5. Clean Test Evaluation
    test_metrics = evaluate(model, test_loader, device, num_classes)
    print(f"==> TEST CLEAN | Macro-F1: {test_metrics['macro_f1']:.4f} | "
          f"BAcc: {test_metrics['balanced_acc']:.4f} | AUROC: {test_metrics['auroc']:.4f} | "
          f"ECE: {test_metrics['ece']:.4f}")

    # 6. Optional Corruption Robustness Evaluation
    corruption_results = {}
    if args.eval_corruptions:
        for corr_name, severities in [("gaussian_noise", [0.05, 0.10, 0.15]), ("gaussian_blur", [5])]:
            for sev in severities:
                _, _, corr_loader, _, _ = build_medmnist_loaders(
                    dataset_name=args.dataset, batch_size=args.batch_size,
                    num_workers=args.num_workers, data_root=args.data_root,
                    test_corruption=corr_name, corruption_severity=sev,
                )
                c_res = evaluate(model, corr_loader, device, num_classes)
                key = f"{corr_name}_{sev}"
                corruption_results[key] = c_res
                print(f"    CORRUPTION [{key}] | Macro-F1: {c_res['macro_f1']:.4f} | AUROC: {c_res['auroc']:.4f}")

    # 7. Save JSON Logs + Spectral Interpretability Profile
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"{args.dataset}_{args.method}_r{args.rank}_K{args.num_bands}_s{args.seed}"

    payload = {
        "config": vars(args),
        "params": param_stats,
        "train_time_sec": train_time_sec,
        "peak_vram_gb": peak_vram_gb,
        "best_val_f1": best_val_f1,
        "test_clean": test_metrics,
        "test_corruptions": corruption_results,
        "spectral_profile": extract_spectral_profile(model) if "sglora" in args.method else {},
    }
    with open(out_dir / f"{run_id}.json", "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved results to {out_dir / f'{run_id}.json'}")


if __name__ == "__main__":
    main()
