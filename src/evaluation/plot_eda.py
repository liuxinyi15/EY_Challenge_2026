#!/usr/bin/env python3
"""
plot_eda.py
===========
读取 data/merged_training_data_clean.csv，为三个目标变量和核心特征自动生成
箱线图 (Boxplots) 和直方图 (Histograms)，保存在 eda_plots/ 文件夹下。

用法：
  python -m src.evaluation.plot_eda
  # 或直接
  python src/evaluation/plot_eda.py
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # 无头后端，适合服务器 / CI
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings("ignore")

# ── 路径 ──────────────────────────────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MERGED_CSV = os.path.join(ROOT, "data", "merged_training_data_clean.csv")
PLOT_DIR = os.path.join(ROOT, "eda_plots")

# ── 目标变量 & 核心特征 ──────────────────────────────────────────────────────
TARGET_COLS = [
    "Total Alkalinity",
    "Electrical Conductance",
    "Dissolved Reactive Phosphorus",
]

CORE_FEATURES = [
    # Landsat 光谱 (API 列名)
    "blue", "green", "red", "nir08", "swir16", "swir22", "NDMI", "MNDWI",
    # TerraClimate 气候
    "pet", "ppt", "tmax", "tmin", "q",
]


def plot_boxplot(series: pd.Series, title: str, save_path: str):
    """绘制单变量箱线图。"""
    fig, ax = plt.subplots(figsize=(8, 3))
    data = series.dropna()
    ax.boxplot(data, vert=False, widths=0.6,
               patch_artist=True,
               boxprops=dict(facecolor="#4C72B0", alpha=0.6),
               medianprops=dict(color="red", linewidth=1.5),
               flierprops=dict(marker="o", markersize=3, alpha=0.4))
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel(title)
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_histogram(series: pd.Series, title: str, save_path: str, bins: int = 60):
    """绘制单变量直方图。"""
    fig, ax = plt.subplots(figsize=(8, 4))
    data = series.dropna()
    ax.hist(data, bins=bins, color="#4C72B0", alpha=0.7, edgecolor="white", linewidth=0.5)
    ax.set_title(f"Distribution: {title}", fontsize=12, fontweight="bold")
    ax.set_xlabel(title)
    ax.set_ylabel("Count")

    # 添加统计摘要
    stats_text = (
        f"n={len(data):,}  mean={data.mean():.2f}  "
        f"median={data.median():.2f}\n"
        f"std={data.std():.2f}  min={data.min():.2f}  max={data.max():.2f}"
    )
    ax.text(0.98, 0.95, stats_text, transform=ax.transAxes,
            fontsize=8, verticalalignment="top", horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5))

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_combined_targets(df: pd.DataFrame, save_path: str):
    """三个目标变量的直方图 + 箱线图合并面板。"""
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(3, 2, width_ratios=[3, 1], hspace=0.35, wspace=0.15)

    for row, col_name in enumerate(TARGET_COLS):
        if col_name not in df.columns:
            continue
        data = df[col_name].dropna()

        # 直方图
        ax_hist = fig.add_subplot(gs[row, 0])
        ax_hist.hist(data, bins=60, color="#4C72B0", alpha=0.7, edgecolor="white", linewidth=0.5)
        ax_hist.set_title(col_name, fontsize=13, fontweight="bold")
        ax_hist.set_ylabel("Count")
        stats_text = f"n={len(data):,}  μ={data.mean():.2f}  σ={data.std():.2f}"
        ax_hist.text(0.98, 0.92, stats_text, transform=ax_hist.transAxes,
                     fontsize=9, ha="right", va="top",
                     bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

        # 箱线图
        ax_box = fig.add_subplot(gs[row, 1])
        ax_box.boxplot(data, vert=True, widths=0.5,
                       patch_artist=True,
                       boxprops=dict(facecolor="#DD8452", alpha=0.6),
                       medianprops=dict(color="red", linewidth=1.5),
                       flierprops=dict(marker="o", markersize=2, alpha=0.3))
        ax_box.set_title("Boxplot", fontsize=11)
        ax_box.set_xticklabels([])

    fig.suptitle("Target Variables — EDA Overview", fontsize=15, fontweight="bold", y=1.01)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ 目标变量面板 → {save_path}")


def plot_feature_grid(df: pd.DataFrame, features: list, save_path: str):
    """核心特征网格直方图。"""
    available = [f for f in features if f in df.columns]
    n = len(available)
    if n == 0:
        return
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows))
    axes = np.array(axes).flatten()

    for i, feat in enumerate(available):
        ax = axes[i]
        data = df[feat].dropna()
        ax.hist(data, bins=50, color="#55A868", alpha=0.7, edgecolor="white", linewidth=0.5)
        ax.set_title(feat, fontsize=11, fontweight="bold")
        ax.set_ylabel("Count")
        ax.text(0.97, 0.92, f"n={len(data):,}\nμ={data.mean():.2f}",
                transform=ax.transAxes, fontsize=8, ha="right", va="top",
                bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.6))

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Core Features — Histograms", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ 特征网格图 → {save_path}")


def plot_correlation_heatmap(df: pd.DataFrame, save_path: str):
    """目标 + 核心特征相关性热力图。"""
    cols = [c for c in TARGET_COLS + CORE_FEATURES if c in df.columns]
    corr = df[cols].corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(cols, fontsize=8)

    # 标注数值
    for i in range(len(cols)):
        for j in range(len(cols)):
            val = corr.iloc[i, j]
            color = "white" if abs(val) > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7, color=color)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Correlation Heatmap", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ 相关性热力图 → {save_path}")


def main():
    if not os.path.exists(MERGED_CSV):
        print(f"❌ 找不到 {MERGED_CSV}，请先运行 build_merged_dataset.py")
        sys.exit(1)

    os.makedirs(PLOT_DIR, exist_ok=True)
    print(f"📂 读取 {MERGED_CSV}…")
    df = pd.read_csv(MERGED_CSV, parse_dates=["Sample Date"])
    print(f"  shape: {df.shape}")

    # ── 1. 目标变量单独图 ──
    for col in TARGET_COLS:
        if col not in df.columns:
            print(f"  ⚠ 列 {col} 不存在，跳过")
            continue
        safe_name = col.replace(" ", "_").lower()
        plot_histogram(df[col], col, os.path.join(PLOT_DIR, f"hist_{safe_name}.png"))
        plot_boxplot(df[col], col, os.path.join(PLOT_DIR, f"box_{safe_name}.png"))
        print(f"  ✅ {col}: histogram + boxplot")

    # ── 2. 目标变量组合面板 ──
    plot_combined_targets(df, os.path.join(PLOT_DIR, "targets_panel.png"))

    # ── 3. 核心特征网格 ──
    plot_feature_grid(df, CORE_FEATURES, os.path.join(PLOT_DIR, "features_grid.png"))

    # ── 4. 核心特征单独箱线图 ──
    for feat in CORE_FEATURES:
        if feat not in df.columns:
            continue
        plot_boxplot(df[feat], feat, os.path.join(PLOT_DIR, f"box_{feat.lower()}.png"))

    # ── 5. 相关性热力图 ──
    plot_correlation_heatmap(df, os.path.join(PLOT_DIR, "correlation_heatmap.png"))

    print(f"\n🎉 所有 EDA 图表已保存至 {PLOT_DIR}/")


if __name__ == "__main__":
    main()
