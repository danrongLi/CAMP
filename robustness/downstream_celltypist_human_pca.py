#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
os.environ["SCIPY_ARRAY_API"] = "1"
os.environ["SKLEARN_ARRAY_API"] = "1"
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("MKL_DYNAMIC", "FALSE")

import json
import time
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import scipy.sparse as sp
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl

import celltypist
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from matplotlib.lines import Line2D


# =========================================================
# Global plot style
# =========================================================
plt.rcParams.update({
    "font.size": 24,
    "axes.titlesize": 28,
    "axes.labelsize": 20,
    "xtick.labelsize": 22,
    "ytick.labelsize": 22,
    "legend.fontsize": 18,
})
plt.rcParams["axes.formatter.use_mathtext"] = True


# =========================================================
# Logging
# =========================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)


# =========================================================
# Constants / style
# =========================================================
CMAP = sns.color_palette("crest", as_cmap=True)

DISPLAY_NAME_MAP = {
    "camp1": "CAMP1",
    "camp2": "CAMP2",
    "camp3": "CAMP3",
    "camp4": "CAMP4",
}

CUSTOM_PALETTE = {
    "CAMP1": "#1f77b4",
    "CAMP2": "#ff7f0e",
    "CAMP3": "#2ca02c",
    "CAMP4": "#bcbd22",
}

VARIANT_ORDER = ["camp1", "camp2", "camp3", "camp4"]

GAMMA_TO_METACELLS = {
    20: 24729,
    30: 16486,
    40: 12364,
    50: 9891,
    60: 8243,
    70: 7065,
    100: 4945,
    150: 3297,
    200: 2472,
    250: 1978,
    300: 1648,
    350: 1413,
    400: 1236,
    450: 1099,
    500: 989,
    550: 899,
}

PCA_LINESTYLES = {
    10: "-",
    15: "--",
    20: "-.",
    30: ":",
    40: (0, (3, 1, 1, 1)),
    50: (0, (5, 1)),
    75: (0, (1, 1)),
    100: (0, (3, 2, 1, 2)),
}


# =========================================================
# Utilities
# =========================================================
def safe_mkdir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def write_metadata_json(path: str, obj: Dict):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def save_txt_list(path: str, values: List[str]):
    with open(path, "w") as f:
        for v in values:
            f.write(f"{v}\n")
    logger.info("Saved txt: %s", path)


def save_df_csv(path: str, df: pd.DataFrame):
    df.to_csv(path, index=False)
    logger.info("Saved csv: %s", path)


def variant_to_display(v: str) -> str:
    return DISPLAY_NAME_MAP.get(v, v)


def variant_to_color(v: str) -> str:
    return CUSTOM_PALETTE.get(variant_to_display(v), "#333333")


def value_to_linestyle(value: int):
    return PCA_LINESTYLES.get(int(value), "-")


def inspect_partition_gamma_columns(part_df: pd.DataFrame, partition_csv: str) -> Dict:
    cols_all = [str(c).strip() for c in part_df.columns]
    cols_seed = [c for c in cols_all if c.endswith("_is_seed")]
    cols_nonseed = [c for c in cols_all if not c.endswith("_is_seed")]

    parsed_gamma_map = {}
    unparsable_nonseed = []

    for c in cols_nonseed:
        try:
            g = int(float(str(c).strip()))
            parsed_gamma_map[g] = c
        except Exception:
            unparsable_nonseed.append(c)

    info = {
        "file": partition_csv,
        "n_total_columns": len(cols_all),
        "n_seed_columns": len(cols_seed),
        "n_nonseed_columns": len(cols_nonseed),
        "parsed_gamma_map": parsed_gamma_map,
        "parsed_gammas_sorted": sorted(parsed_gamma_map.keys()),
        "unparsable_nonseed_columns": unparsable_nonseed,
    }

    logger.info("Partition inspection for %s", partition_csv)
    logger.info("  total columns: %d", info["n_total_columns"])
    logger.info("  seed columns: %d", info["n_seed_columns"])
    logger.info("  non-seed columns: %d", info["n_nonseed_columns"])
    logger.info("  parsed gammas: %s", info["parsed_gammas_sorted"])
    if len(unparsable_nonseed) > 0:
        logger.warning("  unparsable non-seed columns: %s", unparsable_nonseed)

    return info


def previous_get_available_gammas_near_target_n_metacells(
    available_gammas: List[int],
    target_n_metacells: int,
    n_neighbors: int = 3,
) -> List[int]:
    allowed_gammas = [100, 150, 200]
    return [g for g in allowed_gammas if g in available_gammas]

def get_available_gammas_near_target_n_metacells(
    available_gammas: List[int],
    target_n_metacells: int,
    n_neighbors: int = 3,
) -> List[int]:
    available_gammas = [g for g in available_gammas if g in GAMMA_TO_METACELLS]
    return sorted(
        available_gammas,
        key=lambda g: abs(GAMMA_TO_METACELLS[g] - target_n_metacells)
    )[:n_neighbors]


def get_top_k_labels_from_dataset(
    original_adata: ad.AnnData,
    key: str,
    top_k: int = 10,
) -> List[str]:
    label_freq = original_adata.obs[key].astype(str).value_counts()
    return label_freq.head(top_k).index.tolist()


def resolve_preprocessed_h5ad(
    preprocessed_dir: str,
    dataset_name: str,
    variation_value: int,
) -> str:
    candidates = [
        os.path.join(preprocessed_dir, f"{dataset_name}_pca_{variation_value}_prep.h5ad"),
        os.path.join(preprocessed_dir, f"{dataset_name}_camp1_pca_{variation_value}_prep.h5ad"),
        os.path.join(preprocessed_dir, f"{dataset_name}_camp2_pca_{variation_value}_prep.h5ad"),
        os.path.join(preprocessed_dir, f"{dataset_name}_camp3_pca_{variation_value}_prep.h5ad"),
        os.path.join(preprocessed_dir, f"{dataset_name}_camp4_pca_{variation_value}_prep.h5ad"),
    ]
    for p in candidates:
        if os.path.exists(p):
            logger.info("Using preprocessed h5ad: %s", p)
            return p
    raise FileNotFoundError(
        f"Could not find preprocessed h5ad for {dataset_name} pca={variation_value} "
        f"in {preprocessed_dir}. Tried: {candidates}"
    )


def ensure_min_pcs_for_celltypist(
    adata: ad.AnnData,
    min_pcs: int = 50,
) -> ad.AnnData:
    """
    CellTypist majority voting internally constructs a neighbor graph with n_pcs=50.
    For low-PC settings (e.g. 10/15/20/30/40), make sure the prediction object
    still has at least 50 PCs available.
    """
    if adata.n_vars < 2:
        logger.warning("AnnData has fewer than 2 genes; cannot recompute PCA meaningfully.")
        return adata

    need_recompute = (
        "X_pca" not in adata.obsm or
        adata.obsm["X_pca"] is None or
        adata.obsm["X_pca"].shape[1] < min_pcs
    )

    if not need_recompute:
        logger.info(
            "Prediction object already has enough PCs for CellTypist over-clustering: %d",
            adata.obsm["X_pca"].shape[1]
        )
        return adata

    n_comps = min(min_pcs, max(2, adata.n_vars - 1))
    logger.info(
        "Recomputing PCA for CellTypist over-clustering | required_min_pcs=%d | using_n_comps=%d",
        min_pcs, n_comps
    )

    sc.tl.pca(
        adata,
        n_comps=n_comps,
        svd_solver="arpack",
        zero_center=False,
    )

    logger.info("Recomputed X_pca shape=%s", adata.obsm["X_pca"].shape)
    return adata


# =========================================================
# Data helpers
# =========================================================
def create_condense_data(
    partition_df: pd.DataFrame,
    original_adata: ad.AnnData,
    gamma: int,
    key: str,
) -> ad.AnnData:
    col = str(gamma)
    if col not in partition_df.columns:
        raise KeyError(f"Partition column {col} not found.")

    part_col = partition_df[col].dropna().copy()
    part_col.index = part_col.index.astype(str)
    part_col = part_col.astype(str)

    common_cells = original_adata.obs_names.intersection(part_col.index)
    if len(common_cells) == 0:
        raise RuntimeError(f"No overlapping cells between AnnData and partition column {col}.")

    adata_aligned = original_adata[common_cells].copy()
    part_col = part_col.loc[common_cells]

    mc_labels = part_col.values
    unique_mc = pd.Index(pd.unique(mc_labels)).astype(str)

    rows = []
    obs_rows = []
    X = adata_aligned.X
    cell_labels = adata_aligned.obs[key].astype(str).values

    mc_to_label = {}
    for mc in unique_mc:
        idx = np.where(mc_labels == mc)[0]
        X_sub = X[idx]

        if sp.issparse(X_sub):
            mean_vec = np.asarray(X_sub.mean(axis=0)).ravel()
        else:
            mean_vec = np.asarray(X_sub.mean(axis=0)).ravel()

        rows.append(mean_vec)
        obs_rows.append({"metacell_id": mc, "n_cells": len(idx)})

        top_label = pd.Series(cell_labels[idx]).mode().iloc[0]
        mc_to_label[mc] = str(top_label)

    condensed_X = np.vstack(rows).astype(np.float32)
    condensed_obs = pd.DataFrame(obs_rows, index=unique_mc)
    condensed_obs[key] = [mc_to_label[mc] for mc in condensed_obs.index]
    condensed_var = original_adata.var.copy()

    condensed_adata = ad.AnnData(
        X=condensed_X,
        obs=condensed_obs,
        var=condensed_var,
    )

    cols_to_drop = [
        "highly_variable",
        "highly_variable_rank",
        "means",
        "dispersions",
        "dispersions_norm",
        "mean",
        "std",
    ]
    drop_now = [c for c in cols_to_drop if c in condensed_adata.var.columns]
    if len(drop_now) > 0:
        condensed_adata.var = condensed_adata.var.drop(columns=drop_now)

    condensed_adata.raw = condensed_adata.copy()
    return condensed_adata


# =========================================================
# CellTypist
# =========================================================
def run_celltypist_majority_only(
    condensed_adata: ad.AnnData,
    original_adata: ad.AnnData,
    key: str,
    top_labels_fixed: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
    start = time.time()

    if sp.issparse(condensed_adata.X):
        condensed_adata.X = condensed_adata.X.toarray()
    condensed_adata.X = np.asarray(condensed_adata.X, dtype=np.float32)

    if condensed_adata.raw is None:
        condensed_adata.raw = condensed_adata.copy()

    model = celltypist.train(
        condensed_adata,
        key,
        feature_selection=False,
        check_expression=False,
    )
    logger.info("CellTypist train time: %.2fs", time.time() - start)

    start = time.time()
    predictions = celltypist.annotate(
        original_adata,
        model=model,
        majority_voting=True,
        mode="best match",
    )
    logger.info("CellTypist test time: %.2fs", time.time() - start)

    y_true = np.asarray(original_adata.obs[key]).astype(str)
    y_pred = np.asarray(predictions.predicted_labels["majority_voting"]).astype(str).ravel()

    mask = np.isin(y_true, top_labels_fixed)
    y_true_top = y_true[mask]
    y_pred_top = y_pred[mask]

    bal_acc_top = balanced_accuracy_score(y_true_top, y_pred_top)
    cm_top = confusion_matrix(y_true_top, y_pred_top, labels=top_labels_fixed)

    return y_true, y_pred, y_true_top, y_pred_top, bal_acc_top, cm_top


# =========================================================
# Heatmaps + colorbars
# =========================================================
def save_standalone_colorbar(
    vmin: float,
    vmax: float,
    out_base: str,
    dpi: int = 300,
):
    fig, ax = plt.subplots(figsize=(0.9, 6.0))
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cb = mpl.colorbar.ColorbarBase(ax, cmap=CMAP, norm=norm, orientation="vertical")
    cb.ax.tick_params(labelsize=20)

    fig.tight_layout()
    fig.savefig(f"{out_base}.png", dpi=dpi, bbox_inches="tight", transparent=True)
    fig.savefig(f"{out_base}.pdf", bbox_inches="tight", transparent=True)
    plt.close(fig)

    logger.info("Saved colorbar: %s.png", out_base)
    logger.info("Saved colorbar: %s.pdf", out_base)


def plot_confusion_heatmap(
    cm: np.ndarray,
    title: str,
    out_base: str,
    dpi: int = 300,
    fmt: str = ".2f",
):
    plt.close("all")
    fig, ax = plt.subplots(figsize=(12, 10))

    sns.heatmap(
        cm,
        annot=False,
        fmt=fmt,
        cmap=CMAP,
        cbar=False,
        xticklabels=False,
        yticklabels=False,
        ax=ax,
    )

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title(title, fontsize=38, pad=12)

    fig.tight_layout()
    fig.savefig(f"{out_base}.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(f"{out_base}.pdf", bbox_inches="tight")
    plt.close(fig)

    logger.info("Saved heatmap: %s.png", out_base)
    logger.info("Saved heatmap: %s.pdf", out_base)

    save_standalone_colorbar(
        vmin=float(np.min(cm)),
        vmax=float(np.max(cm)),
        out_base=f"{out_base}_colorbar",
        dpi=dpi,
    )


# =========================================================
# Legends for line plots
# =========================================================
def save_overlay_style_legend_png(out_png: str, dpi: int = 600):
    handles = [
        Line2D([0], [0], color="black", linewidth=2.2, alpha=0.25, label="Single setting"),
        Line2D([0], [0], color="black", linewidth=6, marker="o", markersize=10, label="Mean across settings"),
    ]
    fig, ax = plt.subplots(figsize=(4.8, 1.2))
    ax.axis("off")
    ax.legend(handles=handles, loc="center", ncol=2, frameon=False, fontsize=14)
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight", transparent=True)
    plt.close(fig)
    logger.info("Saved overlay style legend: %s", out_png)


def save_near_fixed_metacell_style_legend_png(
    out_png: str,
    target_n_metacells: int,
    dpi: int = 600,
):
    handles = [
        Line2D(
            [0], [0],
            color="black",
            linewidth=3,
            alpha=0.25,
            label=f"Single nearby resolution around {target_n_metacells}",
        ),
        Line2D(
            [0], [0],
            color="black",
            linewidth=6,
            marker="o",
            markersize=10,
            label="Mean across nearby resolutions",
        ),
    ]
    fig, ax = plt.subplots(figsize=(7.5, 1.2))
    ax.axis("off")
    ax.legend(handles=handles, loc="center", ncol=2, frameon=False, fontsize=14)
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight", transparent=True)
    plt.close(fig)
    logger.info("Saved near-fixed-metacell style legend: %s", out_png)


def save_method_legend_png(out_png: str, variants: List[str], dpi: int = 600):
    handles = [
        Line2D([0], [0], color=variant_to_color(v), linewidth=4, marker="o", markersize=8, label=variant_to_display(v))
        for v in variants
    ]
    fig, ax = plt.subplots(figsize=(2.2 * len(handles), 1.2))
    ax.axis("off")
    ax.legend(handles=handles, loc="center", ncol=len(handles), frameon=False, fontsize=14)
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight", transparent=True)
    plt.close(fig)
    logger.info("Saved method legend: %s", out_png)


def save_value_legend_png(out_png: str, values: List[int], dpi: int = 600):
    handles = [
        Line2D(
            [0], [0],
            color="black",
            linestyle=value_to_linestyle(int(v)),
            linewidth=3.5,
            label=f"PC={v}",
        )
        for v in values
    ]
    fig, ax = plt.subplots(figsize=(2.6 * min(len(handles), 4), 1.2))
    ax.axis("off")
    ax.legend(handles=handles, loc="center", ncol=min(len(handles), 4), frameon=False, fontsize=14)
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight", transparent=True)
    plt.close(fig)
    logger.info("Saved value legend: %s", out_png)


# =========================================================
# Line plots
# =========================================================
def plot_balacc_overlay_all_values(
    summary_df: pd.DataFrame,
    out_png: str,
    dpi: int = 300,
):
    if summary_df.empty or "bal_acc_top10_mean" not in summary_df.columns:
        logger.warning("Skipping overlay bal acc plot: missing data")
        return

    df_plot = summary_df.copy()
    fig, ax = plt.subplots(figsize=(9, 6.5))

    variants_present = [v for v in VARIANT_ORDER if v in set(df_plot["variant"])]
    variation_values = sorted(df_plot["variation_value"].unique())

    for variant in variants_present:
        for vv in variation_values:
            ss = df_plot[
                (df_plot["variant"] == variant) &
                (df_plot["variation_value"] == vv)
            ].copy()
            if ss.empty:
                continue

            parsed = []
            for _, row in ss.iterrows():
                gammas = [int(x) for x in str(row["nearby_gammas"]).split(",") if str(x).strip() != ""]
                n_mcs = [GAMMA_TO_METACELLS[g] for g in gammas if g in GAMMA_TO_METACELLS]
                if len(n_mcs) == 0:
                    continue
                xval = float(np.mean(n_mcs))
                parsed.append((xval, float(row["bal_acc_top10_mean"])))

            if len(parsed) == 0:
                continue

            parsed = sorted(parsed, key=lambda x: x[0])
            x = np.array([p[0] for p in parsed], dtype=float)
            y = np.array([p[1] for p in parsed], dtype=float)

            ax.plot(
                x, y,
                color=variant_to_color(variant),
                linewidth=2.2,
                linestyle="-",
                marker=None,
                alpha=0.25,
                zorder=1,
            )

    for variant in variants_present:
        ss = df_plot[df_plot["variant"] == variant].copy()
        if ss.empty:
            continue

        parsed = []
        for _, row in ss.iterrows():
            gammas = [int(x) for x in str(row["nearby_gammas"]).split(",") if str(x).strip() != ""]
            n_mcs = [GAMMA_TO_METACELLS[g] for g in gammas if g in GAMMA_TO_METACELLS]
            if len(n_mcs) == 0:
                continue
            xval = float(np.mean(n_mcs))
            parsed.append((xval, float(row["bal_acc_top10_mean"])))

        if len(parsed) == 0:
            continue

        mean_df = pd.DataFrame(parsed, columns=["n_metacells", "bal_acc"])
        mean_df = mean_df.groupby("n_metacells", as_index=False)["bal_acc"].mean().sort_values("n_metacells")

        ax.plot(
            mean_df["n_metacells"].values.astype(float),
            mean_df["bal_acc"].values.astype(float),
            color=variant_to_color(variant),
            linewidth=6,
            linestyle="-",
            marker="o",
            markersize=10,
            alpha=1.0,
            zorder=3,
        )

    ax.set_xscale("log")
    ax.set_xticks([1e3, 1e4])
    ax.set_xticklabels([r"$10^3$", r"$10^4$"])
    ax.set_xlim(8e2, 3e4)
    ax.set_ylabel("")
    ax.grid(False)
    ax.tick_params(labelsize=28, width=2.0, length=6)

    fig.tight_layout()
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    fig.savefig(out_png.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved overlay plot: %s", out_png)


def plot_balacc_vs_variation_near_fixed_metacells(
    per_gamma_df: pd.DataFrame,
    target_n_metacells: int,
    out_png: str,
    dpi: int = 300,
    n_neighbor_gammas: int = 3,
):
    if per_gamma_df.empty or "bal_acc_top10" not in per_gamma_df.columns:
        logger.warning("Skipping near-fixed bal acc plot: missing data")
        return

    chosen_gammas = sorted(
        GAMMA_TO_METACELLS.keys(),
        key=lambda g: abs(GAMMA_TO_METACELLS[g] - target_n_metacells)
    )[:n_neighbor_gammas]

    sub = per_gamma_df[per_gamma_df["gamma"].isin(chosen_gammas)].copy()
    if sub.empty:
        logger.warning("No rows found near target metacells=%s", target_n_metacells)
        return

    fig, ax = plt.subplots(figsize=(9, 6.5))
    variants_present = [v for v in VARIANT_ORDER if v in set(sub["variant"])]

    for variant in variants_present:
        for gamma in chosen_gammas:
            ss = sub[(sub["variant"] == variant) & (sub["gamma"] == gamma)].copy()
            if ss.empty:
                continue
            ss = ss.sort_values("variation_value")
            ax.plot(
                ss["variation_value"].values.astype(float),
                ss["bal_acc_top10"].values.astype(float),
                color=variant_to_color(variant),
                linewidth=2.2,
                linestyle="-",
                marker=None,
                alpha=0.25,
                zorder=1,
            )

    for variant in variants_present:
        ss = sub[sub["variant"] == variant].copy()
        if ss.empty:
            continue
        mean_df = (
            ss.groupby("variation_value", as_index=False)["bal_acc_top10"]
            .mean()
            .sort_values("variation_value")
        )
        ax.plot(
            mean_df["variation_value"].values.astype(float),
            mean_df["bal_acc_top10"].values.astype(float),
            color=variant_to_color(variant),
            linewidth=6,
            linestyle="-",
            marker="o",
            markersize=10,
            alpha=1.0,
            zorder=3,
        )

    ax.set_ylabel("")
    ax.grid(False)
    ax.tick_params(labelsize=28, width=2.0, length=6)

    fig.tight_layout()
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    fig.savefig(out_png.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved near-fixed-metacell variation plot: %s", out_png)


def make_lineplots_for_downstream(
    per_gamma_df: pd.DataFrame,
    aggregate_df: pd.DataFrame,
    out_dir: str,
    dataset_name: str,
    target_n_metacells: int,
    n_neighbor_gammas: int,
    dpi: int = 300,
):
    if per_gamma_df.empty and aggregate_df.empty:
        logger.warning("Skipping downstream lineplots: no data")
        return

    safe_mkdir(out_dir)

    save_method_legend_png(
        os.path.join(out_dir, f"{dataset_name}_pca_method_legend.png"),
        variants=[v for v in VARIANT_ORDER if v in set(per_gamma_df["variant"]) or v in set(aggregate_df["variant"])],
        dpi=dpi,
    )

    values = []
    if not per_gamma_df.empty:
        values = sorted(per_gamma_df["variation_value"].unique())
    elif not aggregate_df.empty:
        values = sorted(aggregate_df["variation_value"].unique())

    save_value_legend_png(
        os.path.join(out_dir, f"{dataset_name}_pca_value_legend.png"),
        values=[int(v) for v in values],
        dpi=dpi,
    )

    save_overlay_style_legend_png(
        os.path.join(out_dir, f"{dataset_name}_pca_overlay_style_legend.png"),
        dpi=dpi,
    )

    save_near_fixed_metacell_style_legend_png(
        os.path.join(out_dir, f"{dataset_name}_pca_near_fixed_metacell_style_legend.png"),
        target_n_metacells=target_n_metacells,
        dpi=dpi,
    )

    overlay_dir = os.path.join(out_dir, "overlay_metric_plots")
    safe_mkdir(overlay_dir)
    plot_balacc_overlay_all_values(
        summary_df=aggregate_df,
        out_png=os.path.join(overlay_dir, f"{dataset_name}_pca_bal_acc_top10_overlay.png"),
        dpi=dpi,
    )

    near_fixed_dir = os.path.join(out_dir, "metric_vs_feature_near_fixed_metacells")
    safe_mkdir(near_fixed_dir)
    plot_balacc_vs_variation_near_fixed_metacells(
        per_gamma_df=per_gamma_df,
        target_n_metacells=target_n_metacells,
        out_png=os.path.join(
            near_fixed_dir,
            f"{dataset_name}_pca_bal_acc_top10_near_{target_n_metacells}_metacells.png"
        ),
        dpi=dpi,
        n_neighbor_gammas=n_neighbor_gammas,
    )


# =========================================================
# Main downstream worker
# =========================================================
def analyze_one_partition_file(
    original_adata: ad.AnnData,
    original_adata_path: str,
    partition_csv: str,
    dataset_name: str,
    variation_value: int,
    variant: str,
    celltype_key: str,
    top_labels_fixed: List[str],
    target_n_metacells: int,
    n_neighbor_gammas: int,
    out_dir: str,
    dpi: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    part_df = pd.read_csv(partition_csv, index_col=0)
    part_df.columns = [str(c).strip() for c in part_df.columns]

    inspect_info = inspect_partition_gamma_columns(part_df, partition_csv)
    gamma_col_map = inspect_info["parsed_gamma_map"]
    available_gammas = sorted(gamma_col_map.keys())

    chosen_gammas = get_available_gammas_near_target_n_metacells(
        available_gammas=available_gammas,
        target_n_metacells=target_n_metacells,
        n_neighbors=n_neighbor_gammas,
    )

    logger.info("Target metacells: %d", target_n_metacells)
    logger.info("Available gammas in file: %s", available_gammas)
    logger.info(
        "Chosen nearest available gammas: %s (metacells=%s)",
        chosen_gammas,
        [GAMMA_TO_METACELLS[g] for g in chosen_gammas] if len(chosen_gammas) > 0 else [],
    )

    if len(chosen_gammas) == 0:
        logger.warning(
            "Skipping %s | pca=%d | %s because no usable gamma columns found in %s",
            dataset_name, variation_value, variant, partition_csv
        )
        return pd.DataFrame(), pd.DataFrame()

    heatmap_individual_dir = os.path.join(out_dir, "heatmaps_individual")
    heatmap_mean_dir = os.path.join(out_dir, "heatmaps_mean")
    summary_dir = os.path.join(out_dir, "summary")
    label_dir = os.path.join(out_dir, "label_orders")

    for d in [heatmap_individual_dir, heatmap_mean_dir, summary_dir, label_dir]:
        safe_mkdir(d)

    label_txt = os.path.join(
        label_dir,
        f"{dataset_name}_pca_{variation_value}_{variant}_top{len(top_labels_fixed)}_labels_desc.txt"
    )
    if not os.path.exists(label_txt):
        save_txt_list(label_txt, top_labels_fixed)

    per_gamma_rows = []
    cm_list = []
    bal_accs = []

    for gamma in chosen_gammas:
        logger.info(
            "Running downstream | %s | pca=%d | %s | gamma=%d | n_metacells=%d",
            dataset_name, variation_value, variant, gamma, GAMMA_TO_METACELLS[gamma]
        )

        renamed_part_df = part_df.rename(columns={gamma_col_map[gamma]: str(gamma)})

        condensed = create_condense_data(
            partition_df=renamed_part_df,
            original_adata=original_adata,
            gamma=gamma,
            key=celltype_key,
        )

        _, _, y_true_top, y_pred_top, bal_acc_top, cm_top = run_celltypist_majority_only(
            condensed_adata=condensed,
            original_adata=original_adata,
            key=celltype_key,
            top_labels_fixed=top_labels_fixed,
        )

        cm_list.append(cm_top.astype(float))
        bal_accs.append(float(bal_acc_top))

        per_gamma_rows.append({
            "dataset": dataset_name,
            "variation_mode": "pca",
            "variation_value": int(variation_value),
            "variant": variant,
            "gamma": int(gamma),
            "n_metacells": int(GAMMA_TO_METACELLS[gamma]),
            "target_n_metacells": int(target_n_metacells),
            "bal_acc_top10": float(bal_acc_top),
            "n_cells_top10": int(len(y_true_top)),
            "top10_labels_txt": label_txt,
            "partition_csv": partition_csv,
            "original_adata_path": original_adata_path,
        })

        out_base = os.path.join(
            heatmap_individual_dir,
            f"{dataset_name}_pca_{variation_value}_{variant}_gamma{gamma}_top10"
        )
        plot_confusion_heatmap(
            cm=cm_top,
            title=f"{variant_to_display(variant)} (Bal Acc: {bal_acc_top * 100:.2f}%)",
            out_base=out_base,
            dpi=dpi,
            fmt="d",
        )

    mean_cm = np.mean(np.stack(cm_list, axis=0), axis=0)
    mean_bal_acc = float(np.mean(bal_accs))
    std_bal_acc = float(np.std(bal_accs, ddof=0))

    mean_out_base = os.path.join(
        heatmap_mean_dir,
        f"{dataset_name}_pca_{variation_value}_{variant}_mean_near_{target_n_metacells}_top10"
    )
    plot_confusion_heatmap(
        cm=mean_cm,
        title=f"{variant_to_display(variant)} (Mean Bal Acc: {mean_bal_acc * 100:.2f}%)",
        out_base=mean_out_base,
        dpi=dpi,
        fmt=".2f",
    )

    aggregate_df = pd.DataFrame([{
        "dataset": dataset_name,
        "variation_mode": "pca",
        "variation_value": int(variation_value),
        "variant": variant,
        "target_n_metacells": int(target_n_metacells),
        "nearby_gammas": ",".join(str(g) for g in chosen_gammas),
        "nearby_n_metacells": ",".join(str(GAMMA_TO_METACELLS[g]) for g in chosen_gammas),
        "bal_acc_top10_mean": mean_bal_acc,
        "bal_acc_top10_std": std_bal_acc,
        "n_neighbor_gammas": int(len(chosen_gammas)),
        "top10_labels_txt": label_txt,
        "partition_csv": partition_csv,
        "original_adata_path": original_adata_path,
    }])

    return pd.DataFrame(per_gamma_rows), aggregate_df


def run_pca_mode(
    dataset_name: str,
    celltype_key: str,
    partition_dir: str,
    preprocessed_dir: str,
    output_root: str,
    pca_values: List[int],
    variants: List[str],
    target_n_metacells: int,
    n_neighbor_gammas: int,
    top_k: int,
    dpi: int,
):
    logger.info("=" * 80)
    logger.info("Downstream mode: pca")
    logger.info("=" * 80)

    mode_out_dir = os.path.join(output_root, dataset_name, "pca", "downstream_celltypist")
    safe_mkdir(mode_out_dir)

    all_per_gamma = []
    all_aggregate = []

    for variation_value in pca_values:
        try:
            original_adata_path = resolve_preprocessed_h5ad(
                preprocessed_dir=preprocessed_dir,
                dataset_name=dataset_name,
                variation_value=variation_value,
            )
        except FileNotFoundError as e:
            logger.warning("%s", e)
            continue

        logger.info(
            "Loading downstream original_adata | mode=pca | value=%d | path=%s",
            variation_value, original_adata_path
        )
        original_adata = sc.read_h5ad(original_adata_path)
        original_adata.obs_names = original_adata.obs_names.astype(str)

        # Critical patch for CellTypist majority-voting over-clustering
        original_adata = ensure_min_pcs_for_celltypist(original_adata, min_pcs=50)

        if celltype_key not in original_adata.obs.columns:
            logger.warning(
                "Missing celltype key '%s' in %s, skipping pca=%d.",
                celltype_key, original_adata_path, variation_value
            )
            continue

        top_labels_fixed = get_top_k_labels_from_dataset(
            original_adata=original_adata,
            key=celltype_key,
            top_k=top_k,
        )

        logger.info(
            "Top-%d labels | %s | pca=%d: %s",
            top_k, dataset_name, variation_value, top_labels_fixed
        )

        for variant in variants:
            partition_csv = os.path.join(
                partition_dir,
                f"{dataset_name}_{variant}_pca_{variation_value}_partitions.csv"
            )

            if not os.path.exists(partition_csv):
                logger.warning("Missing partition file, skipping: %s", partition_csv)
                continue

            per_gamma_df, aggregate_df = analyze_one_partition_file(
                original_adata=original_adata,
                original_adata_path=original_adata_path,
                partition_csv=partition_csv,
                dataset_name=dataset_name,
                variation_value=variation_value,
                variant=variant,
                celltype_key=celltype_key,
                top_labels_fixed=top_labels_fixed,
                target_n_metacells=target_n_metacells,
                n_neighbor_gammas=n_neighbor_gammas,
                out_dir=mode_out_dir,
                dpi=dpi,
            )

            if not per_gamma_df.empty:
                all_per_gamma.append(per_gamma_df)
            if not aggregate_df.empty:
                all_aggregate.append(aggregate_df)

    per_gamma_all = pd.concat(all_per_gamma, ignore_index=True) if len(all_per_gamma) > 0 else pd.DataFrame()
    aggregate_all = pd.concat(all_aggregate, ignore_index=True) if len(all_aggregate) > 0 else pd.DataFrame()

    summary_dir = os.path.join(mode_out_dir, "summary")
    safe_mkdir(summary_dir)

    per_gamma_csv = os.path.join(summary_dir, f"{dataset_name}_pca_top{top_k}_per_gamma_bal_acc.csv")
    aggregate_csv = os.path.join(summary_dir, f"{dataset_name}_pca_top{top_k}_mean_bal_acc.csv")

    if not per_gamma_all.empty:
        save_df_csv(per_gamma_csv, per_gamma_all)

    if not aggregate_all.empty:
        save_df_csv(aggregate_csv, aggregate_all)

    lineplot_dir = os.path.join(mode_out_dir, "lineplots")
    make_lineplots_for_downstream(
        per_gamma_df=per_gamma_all,
        aggregate_df=aggregate_all,
        out_dir=lineplot_dir,
        dataset_name=dataset_name,
        target_n_metacells=target_n_metacells,
        n_neighbor_gammas=n_neighbor_gammas,
        dpi=dpi,
    )

    run_meta = {
        "dataset_name": dataset_name,
        "variation_mode": "pca",
        "celltype_key": celltype_key,
        "partition_dir": partition_dir,
        "preprocessed_dir": preprocessed_dir,
        "variation_values": [int(v) for v in pca_values],
        "variants": variants,
        "target_n_metacells": int(target_n_metacells),
        "n_neighbor_gammas": int(n_neighbor_gammas),
        "top_k": int(top_k),
        "dpi": int(dpi),
        "celltypist_patch": "ensure X_pca has at least 50 PCs before annotate majority_voting",
        "rule": "use cached preprocessed h5ad from previous variation pipeline for all variants",
    }
    write_metadata_json(
        os.path.join(mode_out_dir, f"{dataset_name}_pca_downstream.meta.json"),
        run_meta,
    )


# =========================================================
# Argparse
# =========================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Human Fetal Atlas downstream CellTypist analysis, PCA only."
    )

    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--celltype_key", type=str, required=True)
    parser.add_argument("--output_root", type=str, required=True)

    parser.add_argument("--pca_partition_dir", type=str, required=True)
    parser.add_argument("--pca_preprocessed_dir", type=str, required=True)

    parser.add_argument("--pca_values", nargs="+", type=int, required=True)

    parser.add_argument("--variants", nargs="+", default=["camp1", "camp2", "camp3", "camp4"])
    parser.add_argument("--target_n_metacells", type=int, default=3000)
    parser.add_argument("--n_neighbor_gammas", type=int, default=3)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--dpi", type=int, default=300)

    return parser.parse_args()


# =========================================================
# Main
# =========================================================
def main():
    args = parse_args()
    logger.info("Arguments:\n%s", json.dumps(vars(args), indent=2))

    if args.celltype_key == "NONE":
        raise ValueError("celltype_key must be a real obs column for downstream analysis.")

    run_pca_mode(
        dataset_name=args.dataset_name,
        celltype_key=args.celltype_key,
        partition_dir=args.pca_partition_dir,
        preprocessed_dir=args.pca_preprocessed_dir,
        output_root=args.output_root,
        pca_values=args.pca_values,
        variants=args.variants,
        target_n_metacells=args.target_n_metacells,
        n_neighbor_gammas=args.n_neighbor_gammas,
        top_k=args.top_k,
        dpi=args.dpi,
    )

    logger.info("Human Fetal Atlas downstream CellTypist PCA-only analysis finished.")


if __name__ == "__main__":
    main()
