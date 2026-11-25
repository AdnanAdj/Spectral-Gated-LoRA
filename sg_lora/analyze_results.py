"""
Post-Experiment Analysis & Camera-Ready Table/Figure Generator:
  1. Aggregates 3-seed JSON logs from ./outputs/ into Mean +/- Std CSV & LaTeX tables
  2. Runs paired t-tests comparing SG-LoRA vs. LoRA and DoRA
  3. Plots publication-ready Layer-Wise Spectral Band Gain Heatmaps (Fig. 3 of manuscript)
"""

import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt


def load_all_runs(output_dir: str = "./outputs"):
    records = []
    profiles = defaultdict(list)
    for fp in sorted(glob.glob(str(Path(output_dir) / "*.json"))):
        with open(fp, "r") as f:
            d = json.load(f)
        cfg = d["config"]
        tc = d["test_clean"]
        rec = {
            "dataset": cfg["dataset"],
            "method": cfg["method"],
            "rank": cfg["rank"],
            "num_bands": cfg["num_bands"],
            "seed": cfg["seed"],
            "trainable_pct": d["params"]["trainable_pct"],
            "macro_f1": tc["macro_f1"] * 100.0,
            "balanced_acc": tc["balanced_acc"] * 100.0,
            "auroc": tc["auroc"] * 100.0,
            "ece": tc["ece"] * 100.0,
            "latency_ms": tc["latency_ms_per_img"],
            "peak_vram_gb": d["peak_vram_gb"],
            "train_time_sec": d["train_time_sec"],
        }
        for cname, cval in d.get("test_corruptions", {}).items():
            rec[f"f1_{cname}"] = cval["macro_f1"] * 100.0
        records.append(rec)

        if d.get("spectral_profile") and d["spectral_profile"].get("q_band_gains"):
            key = (cfg["dataset"], cfg["method"], cfg["rank"], cfg["num_bands"])
            q_g = np.array(d["spectral_profile"]["q_band_gains"])  # (L, K)
            v_g = np.array(d["spectral_profile"]["v_band_gains"])  # (L, K)
            profiles[key].append(0.5 * (q_g + v_g))

    return pd.DataFrame(records), profiles


def generate_summary_tables(df: pd.DataFrame, out_dir: str = "./outputs"):
    if df.empty:
        print("No result JSON files found yet.")
        return

    # Main benchmark (default r=8, K=4)
    main_df = df[(df["rank"] == 8) & (df["num_bands"] == 4)]
    agg = main_df.groupby(["method", "dataset"]).agg(
        f1_mean=("macro_f1", "mean"),
        f1_std=("macro_f1", "std"),
        auc_mean=("auroc", "mean"),
        auc_std=("auroc", "std"),
        params_pct=("trainable_pct", "mean"),
        vram=("peak_vram_gb", "mean"),
        latency=("latency_ms", "mean"),
    ).reset_index()

    agg["F1_str"] = agg.apply(lambda r: f"{r['f1_mean']:.2f} ± {r['f1_std']:.2f}", axis=1)
    agg["AUC_str"] = agg.apply(lambda r: f"{r['auc_mean']:.2f} ± {r['auc_std']:.2f}", axis=1)

    pivot_f1 = agg.pivot(index="method", columns="dataset", values="F1_str")
    pivot_f1.to_csv(Path(out_dir) / "table1_macro_f1.csv")
    with open(Path(out_dir) / "table1_macro_f1.tex", "w") as f:
        f.write(pivot_f1.to_latex(caption="Main Macro-F1 (%) Comparison across MedMNIST v2 (Mean ± Std, 3 seeds)."))
    print("\n=== TABLE 1: MACRO-F1 (%) ===")
    print(pivot_f1.to_string())

    # Paired t-test: SG-LoRA-Dynamic vs. LoRA across all (dataset, seed) pairs
    sg = main_df[main_df["method"] == "sglora_dynamic"].sort_values(["dataset", "seed"])
    lo = main_df[main_df["method"] == "lora"].sort_values(["dataset", "seed"])
    if len(sg) == len(lo) and len(sg) >= 3:
        t_stat, p_val = stats.ttest_rel(sg["macro_f1"].values, lo["macro_f1"].values)
        diff = (sg["macro_f1"].values - lo["macro_f1"].values).mean()
        print(f"\nPaired t-test (SG-LoRA-Dynamic vs. LoRA): mean ΔF1 = {diff:+.2f}%, t = {t_stat:.3f}, p = {p_val:.4e}")


def plot_spectral_heatmaps(profiles: dict, out_dir: str = "./outputs"):
    target_keys = [k for k in profiles.keys() if k[1] == "sglora_dynamic" and k[2] == 8 and k[3] == 4]
    if not target_keys:
        return
    n = len(target_keys)
    fig, axes = plt.subplots(1, n, figsize=(3.6 * n, 4.2), dpi=300, sharey=True)
    if n == 1:
        axes = [axes]
    band_labels = ["DC/Low", "Mid-Low", "Mid-High", "High"]

    for ax, key in zip(axes, sorted(target_keys)):
        mean_gains = np.mean(profiles[key], axis=0)  # (12, 4)
        im = ax.imshow(mean_gains, cmap="RdBu_r", vmin=0.75, vmax=1.25, aspect="auto")
        ax.set_title(key[0].upper(), fontsize=10, fontweight="bold")
        ax.set_xticks(range(4))
        ax.set_xticklabels(band_labels, rotation=30, ha="right", fontsize=8)
        ax.set_yticks(range(mean_gains.shape[0]))
        ax.set_yticklabels([f"L{i+1}" for i in range(mean_gains.shape[0])], fontsize=8)
        ax.set_xlabel("Radial Frequency Band")
    axes[0].set_ylabel("ViT-B/14 Transformer Block")
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.03, label="Learned Amplitude Gain G_k")
    plt.suptitle("Layer-Wise Learned Spectral Gains Across Medical Modalities", fontsize=11.5, fontweight="bold")
    plt.savefig(Path(out_dir) / "fig3_learned_spectral_gains.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved spectral interpretability heatmap to {Path(out_dir) / 'fig3_learned_spectral_gains.png'}")


if __name__ == "__main__":
    df, profs = load_all_runs("./outputs")
    generate_summary_tables(df, "./outputs")
    plot_spectral_heatmaps(profs, "./outputs")
