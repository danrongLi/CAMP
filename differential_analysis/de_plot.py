#!/usr/bin/env python3
"""Render the manuscript DE panels from the original native m=1000 partitions.

This is intentionally a plotting/summary script, not a repartitioning script.
It consumes the cached ``native_partition_stratum_intersection_grid_v2``
checkpoints.  The global native partitions are retained and only intersected
with cell-type x donor x clinical-status boundaries required by the DE task.
No native fragment is merged, split by K-means, reclustered, or forced to a
matched post-stratification count.

Existing matched-10x panels are never overwritten.  All outputs are written to
``native_m1000_original_partitions_v1`` as paired vector PDFs and 350-dpi PNGs,
with the exact plotted values saved in ``figure_data``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/camp_native_m1000_mplconfig")
os.environ.setdefault("NUMBA_CACHE_DIR", "/private/tmp/camp_native_m1000_numba")

import anndata as ad
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.stats import rankdata


HERE = Path(__file__).resolve().parent
RESULT_ROOT = HERE / "results_pbmc_native_fig4_fig5_v2"
NATIVE_DIR: Path
OUTPUT_ROOT: Path
MAIN_DIR: Path
SUPP_DIR: Path
DATA_DIR: Path
FULL_H5AD: Path
PCA_H5AD: Path
COMMON_IDS: Path


def configure_paths(results_root: Path) -> None:
    """Configure every input and output path from one public results root."""
    global RESULT_ROOT, NATIVE_DIR, OUTPUT_ROOT, MAIN_DIR, SUPP_DIR, DATA_DIR
    global FULL_H5AD, PCA_H5AD, COMMON_IDS
    RESULT_ROOT = results_root.expanduser().resolve()
    NATIVE_DIR = (
        RESULT_ROOT
        / "intermediate_csv"
        / "fig5_de_preservation"
        / "native_resolutions"
        / "requested_m1000"
    )
    OUTPUT_ROOT = (
        RESULT_ROOT
        / "figures"
        / "manuscript_additional_experiments_matched_v4"
        / "native_m1000_original_partitions_v1"
    )
    MAIN_DIR = OUTPUT_ROOT / "main_text"
    SUPP_DIR = OUTPUT_ROOT / "supplement"
    DATA_DIR = OUTPUT_ROOT / "figure_data"
    FULL_H5AD = RESULT_ROOT / "cache" / "pbmc_log1p_full_genes.h5ad"
    PCA_H5AD = RESULT_ROOT / "cache" / "pbmc_hvg2000_pca50.h5ad"
    COMMON_IDS = (
        RESULT_ROOT
        / "generated_partitions"
        / "native_resolution_grid"
        / "released_baselines"
        / "common_evaluation_cell_ids.csv.gz"
    )


configure_paths(RESULT_ROOT)

METHOD_FILES: Sequence[Tuple[str, str]] = (
    ("camp1", "CAMP1"),
    ("camp2", "CAMP2"),
    ("camp3", "CAMP3"),
    ("camp4", "CAMP4"),
    ("seacells", "SEACells"),
    ("supercell", "SuperCell"),
    ("metacell1", "MetaCell"),
    ("metacell2", "MetaCell2"),
    ("metaq", "MetaQ"),
)
ALL_METHOD_ORDER = [display for _, display in METHOD_FILES]
DEFAULT_METHOD_ORDER = [
    "CAMP1",
    "SEACells",
    "SuperCell",
    "MetaCell",
    "MetaCell2",
    "MetaQ",
]
DEFAULT_LFC_ORDER = ["Full cells", *DEFAULT_METHOD_ORDER]
ALL_LFC_ORDER = ["Full cells", *ALL_METHOD_ORDER]

PALETTE: Mapping[str, str] = {
    "CAMP1": "#1f77b4",
    "CAMP2": "#ff7f0e",
    "CAMP3": "#2ca02c",
    "CAMP4": "#bcbd22",
    "SEACells": "#d62728",
    "SuperCell": "#9467bd",
    "MetaCell": "#8c564b",
    "MetaCell2": "#e377c2",
    "MetaQ": "#7f7f7f",
}

# Seven broad immune compartments used for the Wilk recurrent-DE analysis.
# The B mapping is the corrected version: granulocyte annotations are not
# assigned to B, while B/plasmablast annotations are.
WILK_MAJOR_CELLTYPES: Sequence[str] = (
    "NK",
    "CD4 T",
    "CD8 T",
    "CD16 Monocyte",
    "DC",
    "CD14 Monocyte",
    "B",
)
WILK_FINE_TO_MAJOR: Mapping[str, str] = {
    "NK": "NK",
    "CD4 T": "CD4 T",
    "CD4m T": "CD4 T",
    "CD4n T": "CD4 T",
    "CD8m T": "CD8 T",
    "CD8eff T": "CD8 T",
    "CD16 Monocyte": "CD16 Monocyte",
    "DC": "DC",
    "pDC": "DC",
    "CD14 Monocyte": "CD14 Monocyte",
    "B": "B",
    "Class-switched B": "B",
    "IgG PB": "B",
    "IgA PB": "B",
}

# Wilk et al. Extended Data Fig. 5 recurrence criterion, translated to the
# retained PBMC matrix.  Selection is derived again from full cells whenever
# this public plotting workflow is run; no hand-curated gene list is required.
WILK_ADJUSTED_P = 0.05
WILK_MIN_ABS_LOGFC = 0.25
WILK_MIN_SAMPLE_SUPPORT = 4
WILK_MIN_PCT = 0.10
PANEL_B_TOP_CELLTYPES = 6
PANEL_B_TOP_GENES_PER_CELLTYPE = 10
STATUS_PALETTE: Mapping[str, str] = {
    "Healthy": "#007C91",
    "COVID": "#E69F00",
}

DPI = 350
AXIS_FONT = 22
TICK_FONT = 18
TITLE_FONT = 24
VALUE_FONT = 15


def set_style() -> None:
    sns.set_theme(style="ticks", context="notebook")
    plt.rcParams.update(
        {
            "font.size": TICK_FONT,
            "axes.titlesize": TITLE_FONT,
            "axes.labelsize": AXIS_FONT,
            "xtick.labelsize": TICK_FONT,
            "ytick.labelsize": TICK_FONT,
            "legend.fontsize": TICK_FONT,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(fig: plt.Figure, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".png"), dpi=DPI, bbox_inches="tight", facecolor="white")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def canonical_method(value: object) -> str:
    value = str(value)
    return {
        "MetaCell1": "MetaCell",
        "Original cells": "Full cells",
    }.get(value, value)


def load_combined_metric(stem: str) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for file_stem, display in METHOD_FILES:
        frame = pd.read_csv(NATIVE_DIR / f"{file_stem}_{stem}.csv")
        frame["method"] = display
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def ranks_within_displayed_genes(
    de_table: pd.DataFrame,
    celltypes: Sequence[str],
    genes: Sequence[str],
) -> pd.DataFrame:
    matrix = (
        de_table.pivot(index="group", columns="names", values="logfoldchanges")
        .reindex(index=list(celltypes), columns=list(genes))
    )
    if matrix.isna().any().any():
        missing = np.argwhere(matrix.isna().to_numpy())
        raise ValueError(f"Cell-type DE display matrix has {len(missing)} missing values")
    ranks: List[np.ndarray] = []
    for _, row in matrix.iterrows():
        ranks.append(rankdata(row.to_numpy(dtype=float), method="average"))
    return pd.DataFrame(ranks, index=matrix.index, columns=matrix.columns)


def displayed_celltype_logfc_from_matrix(
    matrix: np.ndarray,
    labels: np.ndarray,
    celltypes: Sequence[str],
    genes: Sequence[str],
) -> pd.DataFrame:
    """Calculate one-vs-rest approximate log2 fold changes for display genes."""
    rows: List[np.ndarray] = []
    for celltype in celltypes:
        group = labels == str(celltype)
        rest = ~group
        if not group.any() or not rest.any():
            raise ValueError(f"No observations available for cell type {celltype}")
        group_mean = np.asarray(matrix[group], dtype=np.float32).mean(axis=0)
        rest_mean = np.asarray(matrix[rest], dtype=np.float32).mean(axis=0)
        rows.append(
            np.log2(
                (np.expm1(group_mean) + 1e-9)
                / (np.expm1(rest_mean) + 1e-9)
            )
        )
    return pd.DataFrame(rows, index=list(celltypes), columns=list(genes))


def native_display_rank_matrices(
    celltypes: Sequence[str],
    genes: Sequence[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Recreate panel-B ranks from Healthy full cells and native CAMP1 profiles."""
    logfc_matrices, rank_matrices = native_display_rank_matrices_all(celltypes, genes)
    return (
        logfc_matrices["Full cells"],
        logfc_matrices["CAMP1"],
        rank_matrices["Full cells"],
        rank_matrices["CAMP1"],
    )


def native_display_rank_matrices_all(
    celltypes: Sequence[str],
    genes: Sequence[str],
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """Recreate panel-B ranks for full cells and every native-m=1000 method."""
    metadata = load_evaluation_metadata()
    full = ad.read_h5ad(FULL_H5AD, backed="r")
    full_gene_index = pd.Index(full.var_names.astype(str))
    gene_positions = full_gene_index.get_indexer(list(genes))
    if np.any(gene_positions < 0):
        missing = np.asarray(genes, dtype=object)[gene_positions < 0].tolist()
        raise KeyError(f"Panel-B genes missing from the full matrix: {missing}")
    sorted_column_order = np.argsort(gene_positions)
    sorted_positions = gene_positions[sorted_column_order]
    restore_order = np.argsort(sorted_column_order)

    healthy_mask = metadata["Status"].astype(str).to_numpy() == "Healthy"
    full_positions = np.sort(
        metadata.loc[healthy_mask, "matrix_position"].to_numpy(dtype=int)
    )
    full_matrix = np.asarray(full.X[full_positions, :], dtype=np.float32)
    full_matrix = full_matrix[:, sorted_positions][:, restore_order]
    full_labels = full.obs.iloc[full_positions]["cell.type"].astype(str).to_numpy()
    full_lfc = displayed_celltype_logfc_from_matrix(
        full_matrix, full_labels, celltypes, genes
    )

    logfc_matrices: Dict[str, pd.DataFrame] = {"Full cells": full_lfc}
    full_gene_order = np.asarray(full.var_names.astype(str))
    for stem, display in METHOD_FILES:
        metacells = ad.read_h5ad(
            NATIVE_DIR / f"{stem}_native_stratified_metacells.h5ad", backed="r"
        )
        if not np.array_equal(
            np.asarray(metacells.var_names.astype(str)), full_gene_order
        ):
            raise ValueError(
                f"{display} and full-cell matrices do not share the same gene order"
            )
        healthy_rows = np.flatnonzero(
            metacells.obs["condition"].astype(str).to_numpy() == "Healthy"
        )
        matrix = np.asarray(metacells.X[healthy_rows, :], dtype=np.float32)
        matrix = matrix[:, sorted_positions][:, restore_order]
        labels = (
            metacells.obs.iloc[healthy_rows]["celltype"].astype(str).to_numpy()
        )
        logfc_matrices[display] = displayed_celltype_logfc_from_matrix(
            matrix, labels, celltypes, genes
        )

    def rank_matrix(logfc: pd.DataFrame) -> pd.DataFrame:
        rows = [
            rankdata(row.to_numpy(dtype=float), method="average")
            for _, row in logfc.iterrows()
        ]
        return pd.DataFrame(rows, index=logfc.index, columns=logfc.columns)

    rank_matrices = {
        method: rank_matrix(matrix) for method, matrix in logfc_matrices.items()
    }
    return logfc_matrices, rank_matrices


def plot_rank_heatmap_pair(
    full_rank: pd.DataFrame,
    camp1_rank: pd.DataFrame,
    base: Path,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10.8, 8.8), sharex=True)
    fig.subplots_adjust(left=0.17, right=0.99, bottom=0.36, top=0.93, hspace=0.46)
    vmax = full_rank.shape[1]
    for ax, matrix, title in (
        (axes[0], full_rank, "Full data differential expression w.r.t. cell types"),
        (axes[1], camp1_rank, "CAMP1 differential expression w.r.t. cell types"),
    ):
        sns.heatmap(
            matrix,
            cmap=sns.color_palette("Oranges", as_cmap=True),
            vmin=1,
            vmax=vmax,
            cbar=False,
            linewidths=0,
            ax=ax,
        )
        ax.set_title(title, fontsize=TITLE_FONT + 2, pad=11)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=TICK_FONT)
        ax.tick_params(axis="y", length=0, pad=5)
    axes[1].set_xticklabels(
        axes[1].get_xticklabels(), rotation=72, ha="right", fontsize=15
    )
    axes[1].tick_params(axis="x", pad=4)
    save_figure(fig, base)


def plot_rank_heatmap_all_methods(
    rank_matrices: Mapping[str, pd.DataFrame],
    base: Path,
) -> None:
    """Render full cells and all native-m=1000 methods on one shared rank scale."""
    order = ["Full cells", *ALL_METHOD_ORDER]
    missing = [method for method in order if method not in rank_matrices]
    if missing:
        raise ValueError(f"All-method rank grid is missing: {missing}")

    n_cols = 2
    n_rows = int(math.ceil(len(order) / n_cols))
    first = rank_matrices[order[0]]
    vmax = first.shape[1]
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(15.0, 16.8),
        squeeze=False,
    )
    fig.subplots_adjust(
        left=0.085,
        right=0.995,
        bottom=0.145,
        top=0.975,
        wspace=0.075,
        hspace=0.46,
    )
    for panel_index, method in enumerate(order):
        row, column = divmod(panel_index, n_cols)
        ax = axes[row, column]
        sns.heatmap(
            rank_matrices[method],
            cmap=sns.color_palette("Oranges", as_cmap=True),
            vmin=1,
            vmax=vmax,
            cbar=False,
            linewidths=0,
            ax=ax,
        )
        ax.set_title(method, fontsize=TITLE_FONT + 1, pad=9)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(length=0, pad=4)
        if column == 0:
            ax.set_yticklabels(
                ax.get_yticklabels(), rotation=0, fontsize=TICK_FONT - 1
            )
        else:
            ax.set_yticklabels([])
        if row == n_rows - 1:
            ax.set_xticklabels(
                ax.get_xticklabels(), rotation=72, ha="right", fontsize=13
            )
        else:
            ax.set_xticklabels([])
    for panel_index in range(len(order), n_rows * n_cols):
        axes.flat[panel_index].axis("off")
    save_figure(fig, base)


def plot_rank_colorbar(maximum_rank: int, base: Path) -> None:
    fig = plt.figure(figsize=(6.4, 1.05))
    ax = fig.add_axes([0.035, 0.28, 0.93, 0.23])
    colorbar = fig.colorbar(
        ScalarMappable(
            norm=Normalize(vmin=1, vmax=maximum_rank),
            cmap=sns.color_palette("Oranges", as_cmap=True),
        ),
        cax=ax,
        orientation="horizontal",
        ticks=[1, maximum_rank],
    )
    colorbar.outline.set_visible(False)
    colorbar.ax.set_xticklabels(["Low", "High"])
    colorbar.ax.tick_params(labelsize=TICK_FONT, length=0, pad=2)
    colorbar.ax.set_title("Rank", fontsize=AXIS_FONT, pad=-5)
    save_figure(fig, base)


def plot_metric_boxplot(
    frame: pd.DataFrame,
    value_column: str,
    ylabel: str,
    method_order: Sequence[str],
    base: Path,
) -> None:
    work = frame.copy()
    work["method"] = work["method"].map(canonical_method)
    order = [method for method in method_order if method in set(work["method"])]
    work = work[work["method"].isin(order)].copy()
    full_method_panel = len(order) > 6
    width = 5.4 if not full_method_panel else 11.8
    height = 4.8 if not full_method_panel else 6.4
    axis_font = AXIS_FONT if not full_method_panel else 31
    tick_font = TICK_FONT if not full_method_panel else 26
    point_size = 5.6 if not full_method_panel else 7.2
    line_width = 1.8 if not full_method_panel else 2.3
    fig, ax = plt.subplots(figsize=(width, height))
    sns.boxplot(
        data=work,
        x="method",
        y=value_column,
        order=order,
        hue="method",
        hue_order=order,
        palette=PALETTE,
        legend=False,
        width=0.66,
        linewidth=line_width,
        saturation=0.78,
        fliersize=0,
        ax=ax,
    )
    sns.stripplot(
        data=work,
        x="method",
        y=value_column,
        order=order,
        color="#222222",
        size=point_size,
        jitter=0.16,
        alpha=0.58,
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel(ylabel, fontsize=axis_font)
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels(order, rotation=27, ha="right", fontsize=tick_font)
    ax.tick_params(axis="y", labelsize=tick_font, width=line_width)
    ax.tick_params(axis="x", width=line_width)
    low = float(work[value_column].min())
    high = float(work[value_column].max())
    minimum_margin = 0.012 if value_column == "kendall_tau" else 0.004
    margin = max(minimum_margin, 0.08 * (high - low))
    ax.set_ylim(max(-1.0, low - margin), min(1.0, high + margin))
    ax.grid(False)
    sns.despine(ax=ax)
    fig.tight_layout()
    save_figure(fig, base)


def pooled_condition_lfc(matrix: np.ndarray, status: np.ndarray) -> np.ndarray:
    """Compute the same approximate log2 fold change used by the DE pipeline."""
    covid_mean = np.asarray(matrix[status == "COVID"], dtype=np.float32).mean(axis=0)
    healthy_mean = np.asarray(matrix[status == "Healthy"], dtype=np.float32).mean(axis=0)
    return np.log2((np.expm1(covid_mean) + 1e-9) / (np.expm1(healthy_mean) + 1e-9))


def load_evaluation_metadata() -> pd.DataFrame:
    common = pd.read_csv(COMMON_IDS)
    full = ad.read_h5ad(FULL_H5AD, backed="r")
    obs_names = pd.Index(full.obs_names.astype(str))
    positions = obs_names.get_indexer(common["cell_id"].astype(str))
    if np.any(positions < 0):
        raise KeyError("Some common evaluation cells are absent from the full-cell matrix")
    metadata = full.obs.iloc[positions].copy()
    metadata.index = common["cell_id"].astype(str).to_numpy()
    metadata["matrix_position"] = positions
    return metadata


def select_panel_b_celltypes_and_genes() -> Tuple[List[str], List[str]]:
    """Reproduce the MetaQ-style display selection from native full-cell DE."""
    protocol = json.loads((NATIVE_DIR / "de_protocol.json").read_text())
    reference_condition = str(protocol["reference_condition"])
    metadata = load_evaluation_metadata()
    selected_celltypes = (
        metadata.loc[
            metadata["Status"].astype(str) == reference_condition,
            "cell.type",
        ]
        .astype(str)
        .value_counts()
        .head(PANEL_B_TOP_CELLTYPES)
        .index.tolist()
    )
    original_de = pd.read_csv(NATIVE_DIR / "original_celltype_de.csv.gz")
    selected_genes: List[str] = []
    for celltype in selected_celltypes:
        group = original_de[original_de["group"].astype(str) == celltype]
        for gene in group.head(PANEL_B_TOP_GENES_PER_CELLTYPE)["names"].astype(str):
            if gene not in selected_genes:
                selected_genes.append(gene)
    if not selected_celltypes or not selected_genes:
        raise ValueError("The native full-cell DE tables produced no Panel-B selection")
    pd.DataFrame(
        {
            "selection_order": np.arange(1, len(selected_celltypes) + 1),
            "celltype": selected_celltypes,
            "reference_condition": reference_condition,
        }
    ).to_csv(DATA_DIR / "figB_selected_celltypes.csv", index=False)
    pd.DataFrame(
        {
            "selection_order": np.arange(1, len(selected_genes) + 1),
            "gene": selected_genes,
        }
    ).to_csv(DATA_DIR / "figB_selected_genes.csv", index=False)
    return selected_celltypes, selected_genes


def full_status_umap() -> pd.DataFrame:
    """Compute the deterministic all-cell PBMC UMAP used as Panel A."""
    cache = DATA_DIR / "figA_status_umap_all_cells_coordinates.csv.gz"
    if cache.is_file():
        return pd.read_csv(cache)
    if not PCA_H5AD.is_file():
        raise FileNotFoundError(f"The PBMC PCA cache is missing: {PCA_H5AD}")
    reference = ad.read_h5ad(PCA_H5AD)
    if "X_pca" not in reference.obsm:
        raise KeyError(f"X_pca is missing from {PCA_H5AD}")
    for key in ["Status", "cell.type", "Donor"]:
        if key not in reference.obs:
            raise KeyError(f"{key} is missing from {PCA_H5AD}")
    import umap

    reducer = umap.UMAP(
        n_neighbors=15,
        n_components=2,
        metric="euclidean",
        min_dist=0.5,
        spread=1.0,
        random_state=0,
        transform_seed=0,
    )
    coordinates = reducer.fit_transform(np.asarray(reference.obsm["X_pca"]))
    result = pd.DataFrame(
        {
            "item_id": reference.obs_names.astype(str),
            "condition": reference.obs["Status"].astype(str).to_numpy(),
            "celltype": reference.obs["cell.type"].astype(str).to_numpy(),
            "donor": reference.obs["Donor"].astype(str).to_numpy(),
            "UMAP1": coordinates[:, 0],
            "UMAP2": coordinates[:, 1],
        }
    )
    result.to_csv(cache, index=False, compression="gzip")
    return result


def plot_status_umap(frame: pd.DataFrame, base: Path) -> None:
    """Render the all-cell status UMAP with a compact horizontal legend."""
    work = frame.copy()
    work["condition"] = work["condition"].astype(str)
    work = work.sample(frac=1.0, random_state=0)
    order = [value for value in ["Healthy", "COVID"] if value in set(work["condition"])]
    fig, ax = plt.subplots(figsize=(5.35, 5.15))
    fig.subplots_adjust(left=0.015, right=0.985, bottom=0.155, top=0.985)
    ax.scatter(
        work["UMAP1"],
        work["UMAP2"],
        s=3.0,
        alpha=0.72,
        linewidths=0,
        rasterized=True,
        c=work["condition"].map(STATUS_PALETTE).fillna("#777777"),
    )
    ax.set_aspect("equal", adjustable="box")
    ax.margins(x=0.01, y=0.01)
    ax.axis("off")
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=STATUS_PALETTE[condition],
            markeredgecolor="none",
            markersize=6.5,
            label=condition,
        )
        for condition in order
    ]
    ax.legend(
        handles=handles,
        title="Status",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.018),
        ncol=len(handles),
        frameon=False,
        fontsize=TICK_FONT,
        title_fontsize=TICK_FONT + 1,
        markerscale=1.3,
        handletextpad=0.55,
        columnspacing=1.3,
        borderaxespad=0,
    )
    save_figure(fig, base)


def calculate_native_broad_lfc() -> pd.DataFrame:
    cache = DATA_DIR / "native_m1000_all_method_major_celltype_lfc.csv.gz"
    if cache.is_file():
        return pd.read_csv(cache)

    metadata = load_evaluation_metadata()
    full = ad.read_h5ad(FULL_H5AD, backed="r")
    genes = np.asarray(full.var_names.astype(str))
    full_major = metadata["cell.type"].astype(str).map(WILK_FINE_TO_MAJOR)
    frames: List[pd.DataFrame] = []
    for major in WILK_MAJOR_CELLTYPES:
        positions = np.sort(
            metadata.loc[full_major.to_numpy(dtype=object) == major, "matrix_position"]
            .to_numpy(dtype=int)
        )
        if len(positions) == 0:
            raise ValueError(f"Full cells contain no observations for {major}")
        matrix = np.asarray(full.X[positions, :], dtype=np.float32)
        status = full.obs.iloc[positions]["Status"].astype(str).to_numpy()
        frames.append(
            pd.DataFrame(
                {
                    "celltype": major,
                    "names": genes,
                    "method": "Full cells",
                    "logfoldchanges": pooled_condition_lfc(matrix, status),
                }
            )
        )

    for stem, display in METHOD_FILES:
        path = NATIVE_DIR / f"{stem}_native_stratified_metacells.h5ad"
        meta = ad.read_h5ad(path, backed="r")
        meta_genes = np.asarray(meta.var_names.astype(str))
        if not np.array_equal(meta_genes, genes):
            raise ValueError(f"Gene order differs in {path}")
        major_labels = meta.obs["celltype"].astype(str).map(WILK_FINE_TO_MAJOR)
        for major in WILK_MAJOR_CELLTYPES:
            positions = np.flatnonzero(major_labels.to_numpy(dtype=object) == major)
            if len(positions) == 0:
                raise ValueError(f"{display} contains no profiles for {major}")
            matrix = np.asarray(meta.X[positions, :], dtype=np.float32)
            status = meta.obs.iloc[positions]["condition"].astype(str).to_numpy()
            frames.append(
                pd.DataFrame(
                    {
                        "celltype": major,
                        "names": genes,
                        "method": display,
                        "logfoldchanges": pooled_condition_lfc(matrix, status),
                    }
                )
            )

    result = pd.concat(frames, ignore_index=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(cache, index=False, compression="gzip")
    return result


def selected_method_values(all_lfc: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    keys = selected[["celltype", "names", "selection_order", "example"]].copy()
    long = keys.merge(all_lfc, on=["celltype", "names"], how="left")
    expected = len(keys) * len(ALL_LFC_ORDER)
    if len(long) != expected or long["logfoldchanges"].isna().any():
        raise ValueError("The native selected-gene heatmap is incomplete")
    return long


def all_candidate_errors(all_lfc: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    keys = candidates[["celltype", "names"]].drop_duplicates()
    long = keys.merge(all_lfc, on=["celltype", "names"], how="left")
    matrix = long.pivot(
        index=["celltype", "names"], columns="method", values="logfoldchanges"
    ).dropna(subset=ALL_LFC_ORDER)
    rows: List[Dict[str, object]] = []
    for method in ALL_METHOD_ORDER:
        errors = (matrix[method] - matrix["Full cells"]).abs()
        for (celltype, gene), error in errors.items():
            rows.append(
                {
                    "celltype": celltype,
                    "names": gene,
                    "method": method,
                    "absolute_error": float(error),
                }
            )
    return pd.DataFrame(rows)


def derive_wilk_consensus_from_full_cells() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Derive Wilk recurrent genes and display rows without method information."""
    consensus_path = DATA_DIR / "wilk_consensus_gene_celltype_pairs.csv.gz"
    selected_path = DATA_DIR / "selected_consensus_pairs.csv"
    if consensus_path.is_file() and selected_path.is_file():
        return pd.read_csv(consensus_path), pd.read_csv(selected_path)

    metadata = load_evaluation_metadata()
    full = ad.read_h5ad(FULL_H5AD, backed="r")
    genes = np.asarray(full.var_names.astype(str))
    evaluation_major = metadata["cell.type"].astype(str).map(WILK_FINE_TO_MAJOR)
    technical_gene = np.asarray(
        [
            gene.startswith(("RPS", "RPL", "MT-", "MTR", "RNA18S5", "RNA28S5"))
            or gene == "MALAT1"
            for gene in genes
        ],
        dtype=bool,
    )
    sample_frames: List[pd.DataFrame] = []
    for major_celltype in WILK_MAJOR_CELLTYPES:
        major_mask = evaluation_major.to_numpy(dtype=object) == major_celltype
        positions = np.sort(
            metadata.loc[major_mask, "matrix_position"].to_numpy(dtype=int)
        )
        if len(positions) == 0:
            raise ValueError(f"No evaluation cells for Wilk cell type {major_celltype}")
        matrix = np.asarray(full.X[positions, :], dtype=np.float32)
        ordered_obs = full.obs.iloc[positions]
        status = ordered_obs["Status"].astype(str).to_numpy()
        donor = ordered_obs["Donor.full"].astype(str).to_numpy()
        healthy = matrix[status == "Healthy"]
        if healthy.shape[0] == 0:
            raise ValueError(f"No Healthy cells for Wilk cell type {major_celltype}")
        healthy_linear_mean = np.expm1(healthy).mean(axis=0)
        for sample in sorted(np.unique(donor[status == "COVID"]).tolist()):
            case = matrix[(status == "COVID") & (donor == sample)]
            if case.shape[0] == 0:
                continue
            case_linear_mean = np.expm1(case).mean(axis=0)
            # Wilk et al. used Seurat FindMarkers defaults: natural-log average
            # fold change with pseudocount one, Wilcoxon, and Bonferroni P values.
            average_logfc = np.log(
                (case_linear_mean + 1.0) / (healthy_linear_mean + 1.0)
            )
            zscore, pvalue = stats.ranksums(case, healthy, axis=0)
            pvalue = np.asarray(pvalue, dtype=float)
            bonferroni = np.minimum(pvalue * len(genes), 1.0)
            pct_case = np.asarray((case > 0).mean(axis=0), dtype=float)
            pct_healthy = np.asarray((healthy > 0).mean(axis=0), dtype=float)
            frame = pd.DataFrame(
                {
                    "celltype": major_celltype,
                    "names": genes,
                    "covid_sample": sample,
                    "sample_avg_logfc": np.asarray(average_logfc, dtype=float),
                    "sample_wilcoxon_score": np.asarray(zscore, dtype=float),
                    "sample_pvalue_bonferroni": bonferroni,
                    "pct_case": pct_case,
                    "pct_healthy": pct_healthy,
                }
            )
            frame["passes_wilk_rule"] = (
                (np.maximum(frame["pct_case"], frame["pct_healthy"]) >= WILK_MIN_PCT)
                & (~technical_gene)
                & (frame["sample_pvalue_bonferroni"] < WILK_ADJUSTED_P)
                & (frame["sample_avg_logfc"].abs() > WILK_MIN_ABS_LOGFC)
            )
            sample_frames.append(frame)
    if not sample_frames:
        raise ValueError("The Wilk full-cell analysis produced no sample comparisons")
    sample_level = pd.concat(sample_frames, ignore_index=True)
    sample_level["significant_logfc"] = np.where(
        sample_level["passes_wilk_rule"], sample_level["sample_avg_logfc"], 0.0
    )
    consensus = (
        sample_level.groupby(["celltype", "names"], observed=False)
        .agg(
            n_samples_tested=("covid_sample", "nunique"),
            sample_support=("passes_wilk_rule", "sum"),
            cumulative_significant_logfc=("significant_logfc", "sum"),
            mean_sample_logfc=("sample_avg_logfc", "mean"),
            maximum_abs_sample_logfc=(
                "sample_avg_logfc",
                lambda values: float(np.abs(values).max()),
            ),
        )
        .reset_index()
    )
    consensus["is_wilk_consensus"] = (
        consensus["sample_support"] >= WILK_MIN_SAMPLE_SUPPORT
    )
    consensus["abs_cumulative_significant_logfc"] = consensus[
        "cumulative_significant_logfc"
    ].abs()
    consensus.to_csv(consensus_path, index=False, compression="gzip")
    sample_level.drop(columns="significant_logfc").to_csv(
        DATA_DIR / "wilk_sample_level_de.csv.gz", index=False, compression="gzip"
    )

    candidates = consensus[consensus["is_wilk_consensus"].astype(bool)].copy()
    selected_rows: List[pd.Series] = []
    for major_celltype in WILK_MAJOR_CELLTYPES:
        subset = candidates[candidates["celltype"] == major_celltype].sort_values(
            [
                "abs_cumulative_significant_logfc",
                "sample_support",
                "maximum_abs_sample_logfc",
                "names",
            ],
            ascending=[False, False, False, True],
        )
        if subset.empty:
            raise ValueError(f"No recurrent Wilk gene for {major_celltype}")
        selected_rows.append(subset.iloc[0])
    selected = pd.DataFrame(selected_rows).reset_index(drop=True)
    selected["selection_order"] = np.arange(len(selected))
    selected["example"] = selected["celltype"] + " | " + selected["names"]
    selected.to_csv(selected_path, index=False)
    return consensus, selected


def native_wilk_tables() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    all_lfc = calculate_native_broad_lfc()
    consensus, selected = derive_wilk_consensus_from_full_cells()
    selected = selected.sort_values("selection_order").reset_index(drop=True)
    candidates = consensus[consensus["is_wilk_consensus"].astype(bool)].copy()
    selected_long = selected_method_values(all_lfc, selected)
    errors = all_candidate_errors(all_lfc, candidates)
    summary = (
        errors.groupby(["celltype", "method"], observed=False)["absolute_error"]
        .agg(median_absolute_error="median", mean_absolute_error="mean", n_genes="size")
        .reset_index()
    )
    return selected_long, errors, summary


def wilk_matrices(
    selected_long: pd.DataFrame,
    summary: pd.DataFrame,
    lfc_order: Sequence[str],
    error_order: Sequence[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    row_meta = (
        selected_long[["example", "celltype", "selection_order"]]
        .drop_duplicates()
        .sort_values("selection_order")
    )
    examples = row_meta["example"].astype(str).tolist()
    celltypes = row_meta["celltype"].astype(str).tolist()
    lfc_matrix = selected_long.pivot(
        index="example", columns="method", values="logfoldchanges"
    ).reindex(index=examples, columns=list(lfc_order))
    error_matrix = summary.pivot(
        index="celltype", columns="method", values="median_absolute_error"
    ).reindex(index=celltypes, columns=list(error_order))
    if lfc_matrix.isna().any().any() or error_matrix.isna().any().any():
        raise ValueError("Wilk native-m1000 display matrices contain missing values")
    return lfc_matrix, error_matrix


def plot_main_wilk_combined(
    selected_long: pd.DataFrame,
    summary: pd.DataFrame,
    base: Path,
) -> Tuple[float, float]:
    lfc_matrix, error_matrix = wilk_matrices(
        selected_long, summary, DEFAULT_LFC_ORDER, DEFAULT_METHOD_ORDER
    )
    lfc_limit = max(1.0, math.ceil(np.abs(lfc_matrix.to_numpy()).max() * 2) / 2)
    error_limit = max(0.1, math.ceil(error_matrix.to_numpy().max() * 10) / 10)
    fig, (left, right) = plt.subplots(
        1,
        2,
        figsize=(15.2, 6.6),
        gridspec_kw={"width_ratios": [len(DEFAULT_LFC_ORDER), len(DEFAULT_METHOD_ORDER)]},
    )
    fig.subplots_adjust(left=0.19, right=0.99, bottom=0.19, top=0.82, wspace=0.018)
    sns.heatmap(
        lfc_matrix,
        cmap="RdBu_r",
        center=0,
        vmin=-lfc_limit,
        vmax=lfc_limit,
        annot=True,
        fmt=".2f",
        annot_kws={"fontsize": VALUE_FONT},
        linewidths=0,
        cbar=False,
        ax=left,
    )
    sns.heatmap(
        error_matrix,
        cmap="YlOrRd",
        vmin=0,
        vmax=error_limit,
        annot=True,
        fmt=".2f",
        annot_kws={"fontsize": VALUE_FONT},
        linewidths=0,
        cbar=False,
        yticklabels=False,
        ax=right,
    )
    left.set_title("COVID vs Healthy log$_2$ fold changes", fontsize=TITLE_FONT, pad=14)
    right.set_title("Median absolute error across genes", fontsize=TITLE_FONT, pad=14)
    for ax in (left, right):
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(length=0, pad=5)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", fontsize=19)
    left.set_yticklabels(left.get_yticklabels(), rotation=0, fontsize=17)
    save_figure(fig, base)
    return float(lfc_limit), float(error_limit)


def plot_wilk_colorbars(lfc_limit: float, error_limit: float, base: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 1.0))
    fig.subplots_adjust(left=0.03, right=0.98, bottom=0.43, top=0.78, wspace=0.20)
    specifications = (
        (axes[0], "RdBu_r", -lfc_limit, lfc_limit, [-lfc_limit, 0, lfc_limit]),
        (axes[1], "YlOrRd", 0, error_limit, [0, error_limit / 2, error_limit]),
    )
    for ax, cmap, vmin, vmax, ticks in specifications:
        colorbar = fig.colorbar(
            ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap=cmap),
            cax=ax,
            orientation="horizontal",
            ticks=ticks,
        )
        colorbar.outline.set_visible(False)
        colorbar.ax.tick_params(labelsize=16, length=0, pad=2)
    save_figure(fig, base)


def plot_all_method_wilk_combined(
    selected_long: pd.DataFrame,
    summary: pd.DataFrame,
    base: Path,
) -> None:
    """Join the two all-method Wilk heatmaps with aligned rows and colorbars."""
    lfc_matrix, error_matrix = wilk_matrices(
        selected_long,
        summary,
        ALL_LFC_ORDER,
        ALL_METHOD_ORDER,
    )
    lfc_limit = max(1.0, math.ceil(np.abs(lfc_matrix.to_numpy()).max() * 2) / 2)
    error_limit = max(0.1, math.ceil(error_matrix.to_numpy().max() * 10) / 10)

    fig = plt.figure(figsize=(22.0, 9.4))
    # Keep identical cell widths in both matrices and enough central whitespace
    # that the adjacent colorbar endpoint labels never collide.
    left = fig.add_axes([0.15, 0.31, 0.42, 0.58])
    right = fig.add_axes([0.61, 0.31, 0.378, 0.58])
    left_bar = fig.add_axes([0.15, 0.16, 0.42, 0.035])
    right_bar = fig.add_axes([0.61, 0.16, 0.378, 0.035])

    sns.heatmap(
        lfc_matrix,
        cmap="RdBu_r",
        center=0,
        vmin=-lfc_limit,
        vmax=lfc_limit,
        annot=True,
        fmt=".2f",
        annot_kws={"fontsize": 20},
        linewidths=0,
        cbar=False,
        ax=left,
    )
    sns.heatmap(
        error_matrix,
        cmap="YlOrRd",
        vmin=0,
        vmax=error_limit,
        annot=True,
        fmt=".2f",
        annot_kws={"fontsize": 20},
        linewidths=0,
        cbar=False,
        yticklabels=False,
        ax=right,
    )

    left.set_title("COVID vs Healthy log$_2$ fold changes", fontsize=31, pad=16)
    right.set_title("Median absolute error across genes", fontsize=31, pad=16)
    for ax in (left, right):
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(length=0, pad=6)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", fontsize=22)
    left.set_yticklabels(left.get_yticklabels(), rotation=0, fontsize=22)

    left_colorbar = fig.colorbar(
        ScalarMappable(
            norm=Normalize(vmin=-lfc_limit, vmax=lfc_limit), cmap="RdBu_r"
        ),
        cax=left_bar,
        orientation="horizontal",
        ticks=[-lfc_limit, 0, lfc_limit],
    )
    right_colorbar = fig.colorbar(
        ScalarMappable(
            norm=Normalize(vmin=0, vmax=error_limit), cmap="YlOrRd"
        ),
        cax=right_bar,
        orientation="horizontal",
        ticks=[0, error_limit / 2, error_limit],
    )
    for colorbar in (left_colorbar, right_colorbar):
        colorbar.outline.set_visible(False)
        colorbar.ax.tick_params(labelsize=20, length=0, pad=3)
    save_figure(fig, base)


def plot_supp_lfc(selected_long: pd.DataFrame, base: Path) -> float:
    lfc_matrix, _ = wilk_matrices(
        selected_long,
        pd.DataFrame(
            {
                "celltype": [],
                "method": [],
                "median_absolute_error": [],
            }
        ),
        ALL_LFC_ORDER,
        [],
    )
    limit = max(1.0, math.ceil(np.abs(lfc_matrix.to_numpy()).max() * 2) / 2)
    fig, ax = plt.subplots(figsize=(13.2, 7.2))
    sns.heatmap(
        lfc_matrix,
        cmap="RdBu_r",
        center=0,
        vmin=-limit,
        vmax=limit,
        annot=True,
        fmt=".2f",
        annot_kws={"fontsize": VALUE_FONT},
        linewidths=0,
        cbar=False,
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", fontsize=TICK_FONT)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=TICK_FONT)
    ax.tick_params(length=0)
    fig.tight_layout()
    save_figure(fig, base)
    return float(limit)


def plot_supp_error(summary: pd.DataFrame, base: Path) -> float:
    matrix = summary.pivot(
        index="celltype", columns="method", values="median_absolute_error"
    ).reindex(index=WILK_MAJOR_CELLTYPES, columns=ALL_METHOD_ORDER)
    if matrix.isna().any().any():
        raise ValueError("All-method native error heatmap contains missing values")
    limit = max(0.1, math.ceil(matrix.to_numpy().max() * 10) / 10)
    fig, ax = plt.subplots(figsize=(12.7, 7.2))
    sns.heatmap(
        matrix,
        cmap="YlOrRd",
        vmin=0,
        vmax=limit,
        annot=True,
        fmt=".2f",
        annot_kws={"fontsize": VALUE_FONT},
        linewidths=0,
        cbar=False,
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", fontsize=TICK_FONT)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=TICK_FONT)
    ax.tick_params(length=0)
    fig.tight_layout()
    save_figure(fig, base)
    return float(limit)


def write_manifest(
    concordance: pd.DataFrame,
    correlations: pd.DataFrame,
    errors: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    protocol = json.loads((NATIVE_DIR / "de_protocol.json").read_text())
    metacell_summary = pd.read_csv(NATIVE_DIR / "metacell_summary.csv")
    metacell_summary["method"] = metacell_summary["method"].map(canonical_method)
    metacell_summary.to_csv(DATA_DIR / "native_m1000_realized_profile_counts.csv", index=False)

    metric_summary = pd.concat(
        [
            concordance.groupby("method", observed=False)["kendall_tau"]
            .agg(mean="mean", median="median", n="size")
            .reset_index()
            .assign(metric="celltype_de_kendall_tau"),
            correlations.groupby("method", observed=False)["pearson_r"]
            .agg(mean="mean", median="median", n="size")
            .reset_index()
            .assign(metric="condition_de_pearson_r"),
            errors.groupby("method", observed=False)["absolute_error"]
            .agg(mean="mean", median="median", n="size")
            .reset_index()
            .assign(metric="wilk_recurrent_gene_absolute_error"),
        ],
        ignore_index=True,
    )
    metric_summary["method"] = metric_summary["method"].map(canonical_method)
    metric_summary.to_csv(DATA_DIR / "native_m1000_metric_summary.csv", index=False)
    summary.to_csv(DATA_DIR / "native_m1000_wilk_compartment_error_summary.csv", index=False)

    manifest = {
        "protocol": "native_m1000_original_partition_de_figure_package_v1",
        "source_checkpoint_protocol": "native_partition_stratum_intersection_grid_v2",
        "requested_global_metacells": 1000,
        "evaluation_cells": int(protocol["evaluation_cells"]),
        "boundary_operation": "Intersect each native global metacell with celltype x donor x clinical-status boundaries.",
        "not_performed": [
            "post-hoc K-means",
            "native-fragment merging",
            "reclustering",
            "forcing equal realized profile counts",
        ],
        "comparison_note": (
            "This analysis preserves the native partitions nearest global m=1000. "
            "After biological-boundary intersection, realized profile counts differ "
            "across methods and are reported explicitly."
        ),
        "realized_profiles": dict(
            zip(
                metacell_summary["method"],
                metacell_summary["realized_stratified_profiles"].astype(int),
            )
        ),
        "wilk_selection_source": (
            "Recomputed from the common full-cell evaluation data using the "
            "Wilk et al. Extended Data Fig. 5 recurrence criterion; see "
            "figure_data/selected_consensus_pairs.csv."
        ),
        "outputs": {
            "main_text": [
                "figA_status_umap_all_cells_legend_below",
                "figB_native_m1000_de_rank_heatmap_camp1",
                "figB_native_m1000_de_rank_heatmap_all_methods",
                "figB_native_m1000_de_rank_colorbar_horizontal",
                "figC_native_m1000_de_rank_consistency_boxplot",
                "figD_native_m1000_wilk_effects_and_error_camp1",
                "figD_native_m1000_wilk_effects_and_error_all_methods_combined",
                "figD_native_m1000_wilk_colorbars_horizontal",
                "figE_native_m1000_de_condition_consistency_boxplot",
            ],
            "supplement": [
                "figS01_native_m1000_wilk_lfc_all_methods",
                "figS02_native_m1000_wilk_median_error_all_methods",
                "figS01_S02_native_m1000_wilk_colorbars_horizontal",
                "figS03_native_m1000_de_rank_consistency_all_methods",
                "figS04_native_m1000_de_condition_consistency_all_methods",
            ],
        },
    }
    (OUTPUT_ROOT / "figure_manifest.json").write_text(json.dumps(manifest, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render the reproducible native-m=1000 PBMC differential-expression "
            "manuscript and supplementary figures."
        )
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=HERE / "results_pbmc_native_fig4_fig5_v2",
        help="Results directory produced by camp_pbmc_fig4_fig5.py --stage native_de",
    )
    parser.add_argument("--dpi", type=int, default=350)
    return parser.parse_args()


def main() -> None:
    global DPI
    args = parse_args()
    configure_paths(args.results_root)
    DPI = int(args.dpi)
    set_style()
    MAIN_DIR.mkdir(parents=True, exist_ok=True)
    SUPP_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    protocol = json.loads((NATIVE_DIR / "de_protocol.json").read_text())
    if protocol.get("protocol") != "native_partition_stratum_intersection_grid_v2":
        raise ValueError("The source checkpoint is not the expected native-m1000 protocol")
    if int(protocol.get("requested_global_metacells", -1)) != 1000:
        raise ValueError("The source checkpoint does not request global m=1000")

    plot_status_umap(
        full_status_umap(),
        MAIN_DIR / "figA_status_umap_all_cells_legend_below",
    )
    celltypes, genes = select_panel_b_celltypes_and_genes()
    logfc_matrices, rank_matrices = native_display_rank_matrices_all(celltypes, genes)
    filename_stems = {"Full cells": "full_cell"}
    filename_stems.update({display: stem for stem, display in METHOD_FILES})
    for method in ["Full cells", *ALL_METHOD_ORDER]:
        stem = filename_stems[method]
        logfc_matrices[method].to_csv(
            DATA_DIR / f"figB_native_m1000_{stem}_logfc_matrix.csv"
        )
        rank_matrices[method].to_csv(
            DATA_DIR / f"figB_native_m1000_{stem}_rank_matrix.csv"
        )
    plot_rank_heatmap_pair(
        rank_matrices["Full cells"],
        rank_matrices["CAMP1"],
        MAIN_DIR / "figB_native_m1000_de_rank_heatmap_camp1",
    )
    plot_rank_heatmap_all_methods(
        rank_matrices,
        MAIN_DIR / "figB_native_m1000_de_rank_heatmap_all_methods",
    )
    plot_rank_colorbar(
        rank_matrices["Full cells"].shape[1],
        MAIN_DIR / "figB_native_m1000_de_rank_colorbar_horizontal",
    )

    concordance = load_combined_metric("celltype_rank_concordance")
    correlations = load_combined_metric("condition_correlations")
    concordance.to_csv(DATA_DIR / "native_m1000_celltype_rank_concordance.csv", index=False)
    correlations.to_csv(DATA_DIR / "native_m1000_condition_correlations.csv", index=False)
    plot_metric_boxplot(
        concordance,
        "kendall_tau",
        "Kendall's tau",
        DEFAULT_METHOD_ORDER,
        MAIN_DIR / "figC_native_m1000_de_rank_consistency_boxplot",
    )
    plot_metric_boxplot(
        correlations,
        "pearson_r",
        "Pearson correlation",
        DEFAULT_METHOD_ORDER,
        MAIN_DIR / "figE_native_m1000_de_condition_consistency_boxplot",
    )
    plot_metric_boxplot(
        concordance,
        "kendall_tau",
        "Kendall's tau",
        ALL_METHOD_ORDER,
        SUPP_DIR / "figS03_native_m1000_de_rank_consistency_all_methods",
    )
    plot_metric_boxplot(
        correlations,
        "pearson_r",
        "Pearson correlation",
        ALL_METHOD_ORDER,
        SUPP_DIR / "figS04_native_m1000_de_condition_consistency_all_methods",
    )

    selected_long, errors, error_summary = native_wilk_tables()
    selected_long.to_csv(DATA_DIR / "native_m1000_wilk_selected_pair_values.csv", index=False)
    errors.to_csv(DATA_DIR / "native_m1000_wilk_all_candidate_errors.csv.gz", index=False)
    error_summary.to_csv(DATA_DIR / "native_m1000_wilk_compartment_errors.csv", index=False)
    lfc_limit, error_limit = plot_main_wilk_combined(
        selected_long,
        error_summary,
        MAIN_DIR / "figD_native_m1000_wilk_effects_and_error_camp1",
    )
    plot_wilk_colorbars(
        lfc_limit,
        error_limit,
        MAIN_DIR / "figD_native_m1000_wilk_colorbars_horizontal",
    )
    plot_all_method_wilk_combined(
        selected_long,
        error_summary,
        MAIN_DIR / "figD_native_m1000_wilk_effects_and_error_all_methods_combined",
    )
    supplement_lfc_limit = plot_supp_lfc(
        selected_long,
        SUPP_DIR / "figS01_native_m1000_wilk_lfc_all_methods",
    )
    supplement_error_limit = plot_supp_error(
        error_summary,
        SUPP_DIR / "figS02_native_m1000_wilk_median_error_all_methods",
    )
    plot_wilk_colorbars(
        supplement_lfc_limit,
        supplement_error_limit,
        SUPP_DIR / "figS01_S02_native_m1000_wilk_colorbars_horizontal",
    )

    write_manifest(concordance, correlations, errors, error_summary)
    print(f"Rendered native-m1000 DE figure package: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
