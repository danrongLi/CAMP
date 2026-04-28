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
from scipy import sparse
from scipy.sparse import csr_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D

import celltypist
from sklearn.metrics import balanced_accuracy_score, confusion_matrix


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
    20: 1879,
    30: 1253,
    32: 1174,
    35: 1074,
    38: 989,
    42: 895,
    47: 800,
    53: 709,
    60: 626,
    70: 537,
}

SPECIAL_PROCESSED_H5AD = "/storage/home/dvl5760/scratch/blish_covid_processed.h5ad"

SPECIAL_PARTITION_FILES = {
    "camp1": "/storage/home/dvl5760/scratch/partitions_covid_healthy/edit_4_seacell_covid_b_partitions.csv",
    "camp2": "/storage/home/dvl5760/scratch/partitions_covid_healthy/edit_4_seacell_add_simi_covid_b_partitions.csv",
    "camp3": "/storage/home/dvl5760/scratch/partitions_covid_healthy/edit_4_seacell_add_ad_gau_covid_b_partitions.csv",
    "camp4": "/storage/home/dvl5760/scratch/partitions_covid_healthy/edit_5_partitions_full_metacell_covid_healthy.csv",
}

HVG_LINESTYLES = {
    500: "-",
    750: "--",
    1000: "-.",
    1500: ":",
    2000: (0, (3, 1, 1, 1)),
    2500: (0, (5, 1)),
    3000: (0, (1, 1)),
    4000: (0, (3, 2, 1, 2)),
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


def variant_to_display(v: str) -> str:
    return DISPLAY_NAME_MAP.get(v, v)


def variant_to_color(v: str) -> str:
    return CUSTOM_PALETTE.get(variant_to_display(v), "#333333")


def value_to_linestyle(variation_mode: str, value: int):
    if variation_mode == "hvg":
        return HVG_LINESTYLES.get(int(value), "-")
    return PCA_LINESTYLES.get(int(value), "-")


def save_txt_list(path: str, values: List[str]):
    with open(path, "w") as f:
        for v in values:
            f.write(f"{v}\n")
    logger.info("Saved txt: %s", path)


def save_df_csv(path: str, df: pd.DataFrame):
    df.to_csv(path, index=False)
    logger.info("Saved csv: %s", path)


def clip_negative_sparse_inplace(X):
    if sparse.issparse(X):
        if X.nnz > 0:
            X.data = np.maximum(X.data, 0.0)
    else:
        X[:] = np.maximum(X, 0.0)


def ensure_gene_names(adata: ad.AnnData, loom_var_name_key: str = "gene_short_name") -> ad.AnnData:
    if loom_var_name_key in adata.var.columns:
        adata.var_names = adata.var[loom_var_name_key].astype(str)
        adata.var_names_make_unique()
        logger.info("Set var_names from %s", loom_var_name_key)
    else:
        adata.var_names = adata.var_names.astype(str)
        adata.var_names_make_unique()
        logger.info("Using existing var_names")
    adata.var.index.name = None
    return adata


def read_dataset(input_file: str, loom_var_name_key: str = "gene_short_name") -> ad.AnnData:
    ext = Path(input_file).suffix.lower()
    logger.info("Reading dataset: %s", input_file)

    if ext == ".h5ad":
        adata = sc.read_h5ad(input_file)
    elif ext == ".loom":
        adata = sc.read_loom(input_file, sparse=True, dtype="float32")
    else:
        raise ValueError(f"Unsupported input format: {input_file}")

    adata = ensure_gene_names(adata, loom_var_name_key=loom_var_name_key)
    adata.obs_names = adata.obs_names.astype(str)
    return adata


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


def should_use_special_case(variation_mode: str, variation_value: int) -> bool:
    return (
        (variation_mode == "hvg" and int(variation_value) == 2000) or
        (variation_mode == "pca" and int(variation_value) == 50)
    )


def get_partition_csv_for_setting(
    partition_dir: str,
    dataset_name: str,
    variation_mode: str,
    variation_value: int,
    variant: str,
) -> Tuple[str, str]:
    if should_use_special_case(variation_mode, variation_value):
        return SPECIAL_PARTITION_FILES[variant], "special_fixed_partition_csv"

    return (
        os.path.join(
            partition_dir,
            f"{dataset_name}_{variant}_{variation_mode}_{variation_value}_partitions.csv",
        ),
        "standard_partition_csv",
    )


# =========================================================
# Shared non-special downstream input:
# camp4-style preprocessing, STOP BEFORE PCA
# =========================================================
def build_or_load_camp4style_pre_pca_downstream_h5ad(
    input_file: str,
    dataset_name: str,
    output_root: str,
    variation_mode: str,
    variation_value: int,
    fixed_hvg: int,
    fixed_pcs: int,
    loom_var_name_key: str,
    min_genes: int,
    min_cells: int,
) -> str:
    downstream_shared_dir = os.path.join(
        output_root,
        dataset_name,
        variation_mode,
        "downstream_shared_inputs",
    )
    safe_mkdir(downstream_shared_dir)

    out_h5ad = os.path.join(
        downstream_shared_dir,
        f"{dataset_name}_camp4style_shared_{variation_mode}_{variation_value}_prePCA.h5ad"
    )
    meta_json = out_h5ad.replace(".h5ad", ".meta.json")
    tmp_h5ad = out_h5ad + ".tmp"

    if os.path.exists(out_h5ad):
        try:
            _ = sc.read_h5ad(out_h5ad)
            logger.info("Using existing shared downstream prePCA h5ad: %s", out_h5ad)
            return out_h5ad
        except Exception as e:
            logger.warning("Existing shared downstream h5ad is unreadable, rebuilding: %s | error=%s", out_h5ad, e)
            try:
                os.remove(out_h5ad)
            except OSError:
                pass
            if os.path.exists(meta_json):
                try:
                    os.remove(meta_json)
                except OSError:
                    pass
            if os.path.exists(tmp_h5ad):
                try:
                    os.remove(tmp_h5ad)
                except OSError:
                    pass

    if variation_mode == "hvg":
        n_hvg = int(variation_value)
        n_pcs = int(fixed_pcs)
    elif variation_mode == "pca":
        n_hvg = int(fixed_hvg)
        n_pcs = int(variation_value)
    else:
        raise ValueError("variation_mode must be 'hvg' or 'pca'")

    logger.info(
        "Building shared downstream prePCA h5ad | dataset=%s | mode=%s | value=%d | n_hvg=%d | n_pcs=%d",
        dataset_name, variation_mode, variation_value, n_hvg, n_pcs
    )

    adata = read_dataset(input_file, loom_var_name_key=loom_var_name_key)
    logger.info("Raw shape: %s", adata.shape)

    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_genes(adata, min_cells=min_cells)
    logger.info("After filtering: %s", adata.shape)

    clip_negative_sparse_inplace(adata.X)

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    logger.info("Normalized and log1p done")

    if not sparse.issparse(adata.X):
        adata.X = csr_matrix(adata.X)
        logger.info("Converted X to CSR sparse")

    gene_means = np.asarray(adata.X.mean(axis=0)).ravel()
    mask = np.isfinite(gene_means)
    if mask.sum() < adata.shape[1]:
        logger.warning("Dropping %d bad genes", int(adata.shape[1] - mask.sum()))
        adata = adata[:, mask].copy()

    if adata.n_vars == 0:
        raise RuntimeError("Preprocessing left zero genes.")

    n_hvg_use = min(n_hvg, adata.n_vars)
    sc.pp.highly_variable_genes(adata, n_top_genes=n_hvg_use, flavor="seurat")
    if "highly_variable" not in adata.var.columns:
        raise RuntimeError("HVG selection failed in shared downstream preprocessing.")

    logger.info("Selected %d HVGs", int(adata.var["highly_variable"].sum()))
    adata = adata[:, adata.var["highly_variable"]].copy()
    logger.info("After HVG selection: %s", adata.shape)

    # write atomically
    if os.path.exists(tmp_h5ad):
        try:
            os.remove(tmp_h5ad)
        except OSError:
            pass

    adata.write_h5ad(tmp_h5ad, compression="gzip")
    os.replace(tmp_h5ad, out_h5ad)

    # verify file after writing
    _ = sc.read_h5ad(out_h5ad)

    write_metadata_json(meta_json, {
        "dataset": dataset_name,
        "variation_mode": variation_mode,
        "variation_value": int(variation_value),
        "input_file": input_file,
        "loom_var_name_key": loom_var_name_key,
        "min_genes": int(min_genes),
        "min_cells": int(min_cells),
        "n_hvg": int(n_hvg_use),
        "n_pcs_for_reference_only": int(n_pcs),
        "pipeline_note": "camp4-style preprocessing STOPPED before PCA/zero_center=False",
        "n_obs": int(adata.n_obs),
        "n_vars": int(adata.n_vars),
    })
    logger.info("Saved shared downstream prePCA h5ad: %s", out_h5ad)

    return out_h5ad

def previous_build_or_load_camp4style_pre_pca_downstream_h5ad(
    input_file: str,
    dataset_name: str,
    output_root: str,
    variation_mode: str,
    variation_value: int,
    fixed_hvg: int,
    fixed_pcs: int,
    loom_var_name_key: str,
    min_genes: int,
    min_cells: int,
) -> str:
    downstream_shared_dir = os.path.join(
        output_root,
        dataset_name,
        variation_mode,
        "downstream_shared_inputs",
    )
    safe_mkdir(downstream_shared_dir)

    out_h5ad = os.path.join(
        downstream_shared_dir,
        f"{dataset_name}_camp4style_shared_{variation_mode}_{variation_value}_prePCA.h5ad"
    )
    meta_json = out_h5ad.replace(".h5ad", ".meta.json")

    if os.path.exists(out_h5ad):
        logger.info("Using existing shared downstream prePCA h5ad: %s", out_h5ad)
        return out_h5ad

    if variation_mode == "hvg":
        n_hvg = int(variation_value)
        n_pcs = int(fixed_pcs)
    elif variation_mode == "pca":
        n_hvg = int(fixed_hvg)
        n_pcs = int(variation_value)
    else:
        raise ValueError("variation_mode must be 'hvg' or 'pca'")

    logger.info(
        "Building shared downstream prePCA h5ad | dataset=%s | mode=%s | value=%d | n_hvg=%d | n_pcs=%d",
        dataset_name, variation_mode, variation_value, n_hvg, n_pcs
    )

    adata = read_dataset(input_file, loom_var_name_key=loom_var_name_key)
    logger.info("Raw shape: %s", adata.shape)

    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_genes(adata, min_cells=min_cells)
    logger.info("After filtering: %s", adata.shape)

    clip_negative_sparse_inplace(adata.X)

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    logger.info("Normalized and log1p done")

    if not sparse.issparse(adata.X):
        adata.X = csr_matrix(adata.X)
        logger.info("Converted X to CSR sparse")

    gene_means = np.asarray(adata.X.mean(axis=0)).ravel()
    mask = np.isfinite(gene_means)
    if mask.sum() < adata.shape[1]:
        logger.warning("Dropping %d bad genes", int(adata.shape[1] - mask.sum()))
        adata = adata[:, mask].copy()

    if adata.n_vars == 0:
        raise RuntimeError("Preprocessing left zero genes.")

    n_hvg_use = min(n_hvg, adata.n_vars)
    sc.pp.highly_variable_genes(adata, n_top_genes=n_hvg_use, flavor="seurat")
    if "highly_variable" not in adata.var.columns:
        raise RuntimeError("HVG selection failed in shared downstream preprocessing.")

    logger.info("Selected %d HVGs", int(adata.var["highly_variable"].sum()))
    adata = adata[:, adata.var["highly_variable"]].copy()
    logger.info("After HVG selection: %s", adata.shape)

    adata.write_h5ad(out_h5ad, compression="gzip")
    write_metadata_json(meta_json, {
        "dataset": dataset_name,
        "variation_mode": variation_mode,
        "variation_value": int(variation_value),
        "input_file": input_file,
        "loom_var_name_key": loom_var_name_key,
        "min_genes": int(min_genes),
        "min_cells": int(min_cells),
        "n_hvg": int(n_hvg_use),
        "n_pcs_for_reference_only": int(n_pcs),
        "pipeline_note": "camp4-style preprocessing STOPPED before PCA/zero_center=False",
        "n_obs": int(adata.n_obs),
        "n_vars": int(adata.n_vars),
    })
    logger.info("Saved shared downstream prePCA h5ad: %s", out_h5ad)

    return out_h5ad


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


def get_top_k_labels_from_dataset(
    original_adata: ad.AnnData,
    key: str,
    top_k: int = 10,
) -> List[str]:
    label_freq = original_adata.obs[key].astype(str).value_counts()
    return label_freq.head(top_k).index.tolist()


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
# Plotting helpers
# =========================================================
def save_heatmap_colorbar(
    cm: np.ndarray,
    out_base: str,
    dpi: int = 300,
):
    vmin = float(np.min(cm))
    vmax = float(np.max(cm))
    if np.isclose(vmin, vmax):
        vmax = vmin + 1.0

    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    fig, ax = plt.subplots(figsize=(0.8, 6))
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

    save_heatmap_colorbar(cm=cm, out_base=f"{out_base}_colorbar", dpi=dpi)


def save_shared_legend_png(out_png: str, variants: List[str], dpi: int = 600):
    handles = []
    for v in variants:
        handles.append(
            Line2D(
                [0], [0],
                color=variant_to_color(v),
                marker="o",
                linewidth=5.0,
                markersize=10,
                label=variant_to_display(v),
            )
        )

    fig, ax = plt.subplots(figsize=(2.2 * len(handles), 1.2))
    ax.axis("off")
    ax.legend(
        handles=handles,
        loc="center",
        ncol=len(handles),
        frameon=False,
        fontsize=14,
        handlelength=2.0,
        columnspacing=1.4,
    )
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight", transparent=True)
    plt.close(fig)
    logger.info("Saved legend: %s", out_png)


def save_near_fixed_metacell_style_legend_png(
    out_png: str,
    target_n_metacells: int,
    dpi: int = 600,
):
    handles = [
        Line2D(
            [0], [0],
            color="black",
            linewidth=2.2,
            alpha=0.25,
            label=f"Single nearby metacell resolution around {target_n_metacells}",
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
    ax.legend(
        handles=handles,
        loc="center",
        ncol=2,
        frameon=False,
        fontsize=14,
    )
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight", transparent=True)
    plt.close(fig)
    logger.info("Saved near-fixed-metacell style legend: %s", out_png)


def save_method_legend_png(out_png: str, variants: List[str], dpi: int = 600):
    handles = [
        Line2D(
            [0], [0],
            color=variant_to_color(v),
            linewidth=4,
            marker="o",
            markersize=8,
            label=variant_to_display(v),
        )
        for v in variants
    ]

    fig, ax = plt.subplots(figsize=(2.2 * len(handles), 1.2))
    ax.axis("off")
    ax.legend(handles=handles, loc="center", ncol=len(handles), frameon=False, fontsize=14)
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight", transparent=True)
    plt.close(fig)
    logger.info("Saved method legend: %s", out_png)


def save_value_legend_png(out_png: str, variation_mode: str, values: List[int], dpi: int = 600):
    handles = [
        Line2D(
            [0], [0],
            color="black",
            linestyle=value_to_linestyle(variation_mode, int(v)),
            linewidth=3.5,
            label=f"{'HVG' if variation_mode == 'hvg' else 'PC'}={v}",
        )
        for v in values
    ]

    fig, ax = plt.subplots(figsize=(2.6 * min(len(handles), 4), 1.2))
    ax.axis("off")
    ax.legend(
        handles=handles,
        loc="center",
        ncol=min(len(handles), 4),
        frameon=False,
        fontsize=14,
    )
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight", transparent=True)
    plt.close(fig)
    logger.info("Saved value legend: %s", out_png)


def plot_bal_acc_vs_variation_near_fixed_metacells(
    per_gamma_df: pd.DataFrame,
    dataset_name: str,
    variation_mode: str,
    target_n_metacells: int,
    out_png: str,
    dpi: int = 600,
):
    if per_gamma_df.empty or "bal_acc_top10" not in per_gamma_df.columns:
        logger.warning("Skipping downstream line plot because input dataframe is empty.")
        return

    fig, ax = plt.subplots(figsize=(9, 6.5))
    variants_present = [v for v in VARIANT_ORDER if v in set(per_gamma_df["variant"])]

    for variant in variants_present:
        ss_variant = per_gamma_df[per_gamma_df["variant"] == variant].copy()
        if ss_variant.empty:
            continue

        for gamma in sorted(ss_variant["gamma"].dropna().unique()):
            ss = ss_variant[ss_variant["gamma"] == gamma].copy().sort_values("variation_value")
            if ss.empty:
                continue

            x = ss["variation_value"].values.astype(float)
            y = ss["bal_acc_top10"].values.astype(float)

            ax.plot(
                x,
                y,
                color=variant_to_color(variant),
                linewidth=2.2,
                linestyle="-",
                marker=None,
                alpha=0.25,
                zorder=1,
            )

        mean_df = (
            ss_variant.groupby("variation_value", as_index=False)["bal_acc_top10"]
            .mean()
            .sort_values("variation_value")
        )

        x_mean = mean_df["variation_value"].values.astype(float)
        y_mean = mean_df["bal_acc_top10"].values.astype(float)

        ax.plot(
            x_mean,
            y_mean,
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
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_png.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)

    logger.info("Saved downstream line plot: %s", out_png)


def make_downstream_lineplots(
    per_gamma_df: pd.DataFrame,
    out_dir: str,
    dataset_name: str,
    variation_mode: str,
    target_n_metacells: int,
    dpi: int = 600,
):
    if per_gamma_df.empty:
        logger.warning("Skipping downstream lineplots because per_gamma_df is empty.")
        return

    safe_mkdir(out_dir)

    save_method_legend_png(
        os.path.join(out_dir, f"{dataset_name}_{variation_mode}_method_legend.png"),
        variants=[v for v in VARIANT_ORDER if v in set(per_gamma_df["variant"])],
        dpi=dpi,
    )

    variation_values = sorted(per_gamma_df["variation_value"].dropna().unique().tolist())
    save_value_legend_png(
        os.path.join(out_dir, f"{dataset_name}_{variation_mode}_value_legend.png"),
        variation_mode=variation_mode,
        values=[int(v) for v in variation_values],
        dpi=dpi,
    )

    save_near_fixed_metacell_style_legend_png(
        os.path.join(out_dir, f"{dataset_name}_{variation_mode}_near_fixed_metacell_style_legend.png"),
        target_n_metacells=target_n_metacells,
        dpi=dpi,
    )

    out_png = os.path.join(
        out_dir,
        f"{dataset_name}_{variation_mode}_bal_acc_top10_near_{target_n_metacells}_metacells.png"
    )
    plot_bal_acc_vs_variation_near_fixed_metacells(
        per_gamma_df=per_gamma_df,
        dataset_name=dataset_name,
        variation_mode=variation_mode,
        target_n_metacells=target_n_metacells,
        out_png=out_png,
        dpi=dpi,
    )


# =========================================================
# Main downstream driver
# =========================================================
def analyze_one_partition_file(
    original_adata: ad.AnnData,
    original_adata_path: str,
    partition_csv: str,
    partition_source_tag: str,
    dataset_name: str,
    variation_mode: str,
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
            "Skipping %s | %s=%d | %s because no usable gamma columns found in %s",
            dataset_name, variation_mode, variation_value, variant, partition_csv
        )
        return pd.DataFrame(), pd.DataFrame()

    logger.info(
        "Analyzing %s | %s=%d | %s | chosen_gammas=%s | partition_source=%s",
        dataset_name, variation_mode, variation_value, variant, chosen_gammas, partition_source_tag
    )

    heatmap_individual_dir = os.path.join(out_dir, "heatmaps_individual")
    heatmap_mean_dir = os.path.join(out_dir, "heatmaps_mean")
    summary_dir = os.path.join(out_dir, "summary")
    label_dir = os.path.join(out_dir, "label_orders")

    for d in [heatmap_individual_dir, heatmap_mean_dir, summary_dir, label_dir]:
        safe_mkdir(d)

    label_txt = os.path.join(
        label_dir,
        f"{dataset_name}_{variation_mode}_{variation_value}_{variant}_top{len(top_labels_fixed)}_labels_desc.txt"
    )
    if not os.path.exists(label_txt):
        save_txt_list(label_txt, top_labels_fixed)

    per_gamma_rows = []
    cm_list = []
    bal_accs = []

    for gamma in chosen_gammas:
        logger.info(
            "Running downstream | %s | %s=%d | %s | gamma=%d | n_metacells=%d",
            dataset_name, variation_mode, variation_value, variant, gamma, GAMMA_TO_METACELLS[gamma]
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
            "variation_mode": variation_mode,
            "variation_value": int(variation_value),
            "variant": variant,
            "gamma": int(gamma),
            "n_metacells": int(GAMMA_TO_METACELLS[gamma]),
            "target_n_metacells": int(target_n_metacells),
            "bal_acc_top10": float(bal_acc_top),
            "n_cells_top10": int(len(y_true_top)),
            "top10_labels_txt": label_txt,
            "partition_csv": partition_csv,
            "partition_source_tag": partition_source_tag,
            "original_adata_path": original_adata_path,
        })

        out_base = os.path.join(
            heatmap_individual_dir,
            f"{dataset_name}_{variation_mode}_{variation_value}_{variant}_gamma{gamma}_top10"
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
        f"{dataset_name}_{variation_mode}_{variation_value}_{variant}_mean_near_{target_n_metacells}_top10"
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
        "variation_mode": variation_mode,
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
        "partition_source_tag": partition_source_tag,
        "original_adata_path": original_adata_path,
    }])

    return pd.DataFrame(per_gamma_rows), aggregate_df


def run_one_mode(
    input_h5ad: str,
    output_root: str,
    dataset_name: str,
    celltype_key: str,
    partition_dir: str,
    variation_mode: str,
    variation_values: List[int],
    fixed_hvg: int,
    fixed_pcs: int,
    variants: List[str],
    target_n_metacells: int,
    n_neighbor_gammas: int,
    top_k: int,
    dpi: int,
    loom_var_name_key: str,
    min_genes: int,
    min_cells: int,
):
    logger.info("=" * 80)
    logger.info("Downstream mode: %s", variation_mode)
    logger.info("=" * 80)

    mode_out_dir = os.path.join(output_root, dataset_name, variation_mode, "downstream_celltypist")
    safe_mkdir(mode_out_dir)

    all_per_gamma = []
    all_aggregate = []

    for variation_value in variation_values:
        is_special = should_use_special_case(variation_mode, variation_value)

        if is_special:
            original_adata_path = SPECIAL_PROCESSED_H5AD
            if not os.path.exists(original_adata_path):
                logger.warning("Missing special processed h5ad, skipping special setting: %s", original_adata_path)
                continue
        else:
            original_adata_path = build_or_load_camp4style_pre_pca_downstream_h5ad(
                input_file=input_h5ad,
                dataset_name=dataset_name,
                output_root=output_root,
                variation_mode=variation_mode,
                variation_value=variation_value,
                fixed_hvg=fixed_hvg,
                fixed_pcs=fixed_pcs,
                loom_var_name_key=loom_var_name_key,
                min_genes=min_genes,
                min_cells=min_cells,
            )

        logger.info(
            "Loading downstream original_adata | mode=%s | value=%d | special=%s | path=%s",
            variation_mode, variation_value, is_special, original_adata_path
        )
        original_adata = sc.read_h5ad(original_adata_path)
        original_adata.obs_names = original_adata.obs_names.astype(str)

        if celltype_key not in original_adata.obs.columns:
            logger.warning("Missing celltype key '%s' in %s, skipping value=%d.", celltype_key, original_adata_path, variation_value)
            continue

        top_labels_fixed = get_top_k_labels_from_dataset(
            original_adata=original_adata,
            key=celltype_key,
            top_k=top_k,
        )

        for variant in variants:
            partition_csv, partition_source_tag = get_partition_csv_for_setting(
                partition_dir=partition_dir,
                dataset_name=dataset_name,
                variation_mode=variation_mode,
                variation_value=variation_value,
                variant=variant,
            )

            if not os.path.exists(partition_csv):
                logger.warning("Missing partition file, skipping: %s", partition_csv)
                continue

            logger.info(
                "Top-%d labels | %s | %s=%d | %s: %s",
                top_k, dataset_name, variation_mode, variation_value, variant, top_labels_fixed
            )

            per_gamma_df, aggregate_df = analyze_one_partition_file(
                original_adata=original_adata,
                original_adata_path=original_adata_path,
                partition_csv=partition_csv,
                partition_source_tag=partition_source_tag,
                dataset_name=dataset_name,
                variation_mode=variation_mode,
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

    if not per_gamma_all.empty:
        per_gamma_csv = os.path.join(summary_dir, f"{dataset_name}_{variation_mode}_top{top_k}_per_gamma_bal_acc.csv")
        save_df_csv(per_gamma_csv, per_gamma_all)

        lineplot_dir = os.path.join(mode_out_dir, "lineplots")
        safe_mkdir(lineplot_dir)
        make_downstream_lineplots(
            per_gamma_df=per_gamma_all,
            out_dir=lineplot_dir,
            dataset_name=dataset_name,
            variation_mode=variation_mode,
            target_n_metacells=target_n_metacells,
            dpi=dpi,
        )

    if not aggregate_all.empty:
        mean_csv = os.path.join(summary_dir, f"{dataset_name}_{variation_mode}_top{top_k}_mean_bal_acc.csv")
        save_df_csv(mean_csv, aggregate_all)

    run_meta = {
        "dataset_name": dataset_name,
        "variation_mode": variation_mode,
        "celltype_key": celltype_key,
        "partition_dir": partition_dir,
        "variation_values": [int(v) for v in variation_values],
        "fixed_hvg": int(fixed_hvg),
        "fixed_pcs": int(fixed_pcs),
        "variants": variants,
        "target_n_metacells": int(target_n_metacells),
        "n_neighbor_gammas": int(n_neighbor_gammas),
        "top_k": int(top_k),
        "dpi": int(dpi),
        "special_partition_files": SPECIAL_PARTITION_FILES,
        "special_processed_h5ad": SPECIAL_PROCESSED_H5AD,
        "special_rule": "for hvg=2000 or pca=50, use special processed h5ad + special partition csvs",
        "non_special_rule": "for all other settings, use shared camp4-style preprocessing STOPPED before PCA",
        "lineplot_rule": "faint nearby gamma lines and thick mean-across-nearby lines are saved under lineplots/",
        "heatmap_colorbar_rule": "each heatmap is saved without colorbar, and a matching standalone colorbar png/pdf is saved separately",
    }
    write_metadata_json(
        os.path.join(mode_out_dir, f"{dataset_name}_{variation_mode}_downstream.meta.json"),
        run_meta,
    )


# =========================================================
# Argparse
# =========================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="PBMC downstream CellTypist analysis."
    )

    parser.add_argument("--input_h5ad", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--celltype_key", type=str, required=True)
    parser.add_argument("--output_root", type=str, required=True)

    parser.add_argument("--hvg_partition_dir", type=str, required=True)
    parser.add_argument("--pca_partition_dir", type=str, required=True)

    parser.add_argument("--hvg_values", nargs="+", type=int, required=True)
    parser.add_argument("--pca_values", nargs="+", type=int, required=True)

    parser.add_argument("--fixed_hvg", type=int, default=2000)
    parser.add_argument("--fixed_pcs", type=int, default=50)

    parser.add_argument("--variants", nargs="+", default=["camp1", "camp2", "camp3", "camp4"])
    parser.add_argument("--target_n_metacells", type=int, default=1000)
    parser.add_argument("--n_neighbor_gammas", type=int, default=3)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--dpi", type=int, default=300)

    parser.add_argument("--loom_var_name_key", type=str, default="gene_short_name")
    parser.add_argument("--min_genes", type=int, default=200)
    parser.add_argument("--min_cells", type=int, default=3)

    return parser.parse_args()


# =========================================================
# Main
# =========================================================
def main():
    args = parse_args()
    logger.info("Arguments:\n%s", json.dumps(vars(args), indent=2))

    if args.dataset_name == "PBMC":
        args.hvg_values = [v for v in args.hvg_values if int(v) != 4000]

    if args.celltype_key == "NONE":
        raise ValueError("celltype_key must be a real obs column for downstream analysis.")

    if not os.path.exists(args.input_h5ad):
        raise FileNotFoundError(f"Missing input_h5ad: {args.input_h5ad}")

    run_one_mode(
        input_h5ad=args.input_h5ad,
        output_root=args.output_root,
        dataset_name=args.dataset_name,
        celltype_key=args.celltype_key,
        partition_dir=args.hvg_partition_dir,
        variation_mode="hvg",
        variation_values=args.hvg_values,
        fixed_hvg=args.fixed_hvg,
        fixed_pcs=args.fixed_pcs,
        variants=args.variants,
        target_n_metacells=args.target_n_metacells,
        n_neighbor_gammas=args.n_neighbor_gammas,
        top_k=args.top_k,
        dpi=args.dpi,
        loom_var_name_key=args.loom_var_name_key,
        min_genes=args.min_genes,
        min_cells=args.min_cells,
    )

    run_one_mode(
        input_h5ad=args.input_h5ad,
        output_root=args.output_root,
        dataset_name=args.dataset_name,
        celltype_key=args.celltype_key,
        partition_dir=args.pca_partition_dir,
        variation_mode="pca",
        variation_values=args.pca_values,
        fixed_hvg=args.fixed_hvg,
        fixed_pcs=args.fixed_pcs,
        variants=args.variants,
        target_n_metacells=args.target_n_metacells,
        n_neighbor_gammas=args.n_neighbor_gammas,
        top_k=args.top_k,
        dpi=args.dpi,
        loom_var_name_key=args.loom_var_name_key,
        min_genes=args.min_genes,
        min_cells=args.min_cells,
    )

    logger.info("Downstream CellTypist analysis finished.")


if __name__ == "__main__":
    main()
