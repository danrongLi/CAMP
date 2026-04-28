#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import logging
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("MKL_DYNAMIC", "FALSE")


# =========================================================
# Logging
# =========================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
logger = logging.getLogger(__name__)


# =========================================================
# Style
# =========================================================
DISPLAY_NAME_MAP = {
    "camp1": "CAMP1",
    "camp2": "CAMP2",
    "camp3": "CAMP3",
    "camp4": "CAMP4",
    "seacells": "SEACells",
    "supercell": "SuperCell",
    "metacell1": "MetaCell",
    "metacell2": "MetaCell2",
    "metaq": "MetaQ",
}

CUSTOM_PALETTE = {
    "CAMP1": "#1f77b4",
    "CAMP2": "#ff7f0e",
    "CAMP3": "#2ca02c",
    "SEACells": "#d62728",
    "SuperCell": "#9467bd",
    "MetaCell": "#8c564b",
    "MetaCell2": "#e377c2",
    "MetaQ": "#7f7f7f",
    "CAMP4": "#bcbd22",
}

METHOD_ORDER = [
    "camp1",
    "camp2",
    "camp3",
    "camp4",
    "seacells",
    "supercell",
    "metacell1",
    "metacell2",
    "metaq",
]

PRETTY_METRIC_LABELS = {
    "rare_recall": "Recall",
    "fragmentation_per_cell": "Fragmentation",
    "rare_entropy": "Entropy",
    "rare_precision": "Precision",
    "rare_f1": "F1",
    "mean_metacell_purity": "Mean Metacell Purity",
}

CORE_METRICS_TO_PLOT = [
    "rare_recall",
    "fragmentation_per_cell",
    "rare_entropy",
    "rare_precision",
    "rare_f1",
    "mean_metacell_purity",
]


# =========================================================
# Helpers
# =========================================================
def safe_mkdir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def variant_to_display(v: str) -> str:
    return DISPLAY_NAME_MAP.get(str(v).lower(), str(v))


def variant_to_color(v: str) -> str:
    return CUSTOM_PALETTE.get(variant_to_display(v), "#333333")


def save_shared_legend_png(out_png: str, methods: List[str], dpi: int = 600):
    handles = []
    for m in methods:
        handles.append(
            Line2D(
                [0], [0],
                color=variant_to_color(m),
                marker="o",
                linewidth=2.2,
                markersize=6,
                label=variant_to_display(m),
            )
        )

    fig, ax = plt.subplots(figsize=(2.2 * max(1, len(handles)), 1.2))
    ax.axis("off")
    ax.legend(
        handles=handles,
        loc="center",
        ncol=max(1, len(handles)),
        frameon=False,
        fontsize=12,
        handlelength=2.0,
        columnspacing=1.4,
    )
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight", transparent=True)
    plt.close(fig)
    logger.info("Saved legend: %s", out_png)


def save_rare_cell_order_txt(
    rare_type_df: pd.DataFrame,
    rare_threshold: float,
    out_txt: str,
):
    rare_sub = rare_type_df[rare_type_df["rare_threshold"] == rare_threshold].copy()
    rare_sub = rare_sub.sort_values(["n_cells", "cell_type"], ascending=[False, True])

    with open(out_txt, "w") as f:
        for _, row in rare_sub.iterrows():
            ct = str(row["cell_type"])
            if ct == "Class-switched B":
                continue
            f.write(f"{ct}\t{row['n_cells']}\n")

    logger.info("Saved rare cell order: %s", out_txt)


def get_selected_resolution_table(
    raw_df: pd.DataFrame,
    target_n_metacells: int,
    rare_threshold: float,
    n_nearest: int = 3,
) -> Tuple[pd.DataFrame, Dict[str, List[int]]]:
    """
    Returns:
      1) a table with one chosen center resolution per method (nearest to target)
      2) a dict method -> list of up to n_nearest n_metacells values nearest to target
    """
    sub = raw_df[raw_df["rare_threshold"] == rare_threshold].copy()
    if sub.empty:
        raise ValueError(f"No raw rows found for rare_threshold={rare_threshold}")

    selection_rows = []
    method_to_nearest_n = {}

    for method in METHOD_ORDER:
        ss = sub[sub["method"] == method].copy()
        if ss.empty:
            continue

        unique_res = ss[["method", "gamma", "n_metacells", "rare_threshold"]].drop_duplicates().copy()
        unique_res["abs_diff_to_target"] = (unique_res["n_metacells"] - target_n_metacells).abs()
        unique_res = unique_res.sort_values(
            by=["abs_diff_to_target", "n_metacells", "gamma"],
            ascending=[True, True, True]
        ).reset_index(drop=True)

        center_row = unique_res.iloc[0].copy()
        selection_rows.append(center_row)

        nearest_rows = unique_res.head(n_nearest).copy()
        method_to_nearest_n[method] = nearest_rows["n_metacells"].astype(int).tolist()

    if len(selection_rows) == 0:
        raise ValueError("No methods available after filtering.")

    selected_table = pd.DataFrame(selection_rows).reset_index(drop=True)
    selected_table["display_name"] = selected_table["method"].map(variant_to_display)
    return selected_table, method_to_nearest_n


def save_selection_table(selection_df: pd.DataFrame, out_csv: str):
    cols = [
        "method",
        "display_name",
        "gamma",
        "n_metacells",
        "abs_diff_to_target",
        "rare_threshold",
    ]
    keep_cols = [c for c in cols if c in selection_df.columns]
    selection_df[keep_cols].drop_duplicates().to_csv(out_csv, index=False)
    logger.info("Saved selection table: %s", out_csv)


def save_nearest_resolution_lists(
    selection_df: pd.DataFrame,
    method_to_nearest_n: Dict[str, List[int]],
    out_csv: str,
):
    rows = []
    for _, row in selection_df.iterrows():
        method = row["method"]
        nearest_list = method_to_nearest_n.get(method, [])
        rows.append({
            "method": method,
            "display_name": variant_to_display(method),
            "center_gamma": row["gamma"],
            "center_n_metacells": row["n_metacells"],
            "nearest_n_metacells_list": ",".join(str(x) for x in nearest_list),
        })
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    logger.info("Saved nearest-resolution list table: %s", out_csv)


def build_center_and_interval_df(
    raw_df: pd.DataFrame,
    rare_threshold: float,
    selection_df: pd.DataFrame,
    method_to_nearest_n: Dict[str, List[int]],
    metric_cols: List[str],
) -> pd.DataFrame:
    """
    For each method and rare cell type:
      - center value = metric at the single nearest resolution to target
      - interval = min/max across the 3 nearest resolutions
    """
    sub = raw_df[raw_df["rare_threshold"] == rare_threshold].copy()
    rows = []

    for _, sel in selection_df.iterrows():
        method = sel["method"]
        center_n = int(sel["n_metacells"])
        center_gamma = int(sel["gamma"])
        nearest_n_list = method_to_nearest_n.get(method, [])

        method_all = sub[sub["method"] == method].copy()
        method_center = method_all[
            (method_all["n_metacells"] == center_n) &
            (method_all["gamma"] == center_gamma)
        ].copy()

        method_near = method_all[method_all["n_metacells"].isin(nearest_n_list)].copy()

        rare_types = sorted(set(method_center["cell_type"].astype(str)))

        for ct in rare_types:
            row = {
                "method": method,
                "display_name": variant_to_display(method),
                "cell_type": ct,
                "center_n_metacells": center_n,
                "center_gamma": center_gamma,
                "nearest_n_metacells_list": ",".join(str(x) for x in nearest_n_list),
            }

            center_ct = method_center[method_center["cell_type"].astype(str) == ct]
            near_ct = method_near[method_near["cell_type"].astype(str) == ct]

            if center_ct.empty:
                continue

            for metric in metric_cols:
                if metric not in method_all.columns:
                    continue

                center_val = float(center_ct.iloc[0][metric])

                vals = near_ct[metric].dropna().astype(float).values
                if len(vals) == 0:
                    low_val = np.nan
                    high_val = np.nan
                else:
                    low_val = float(np.min(vals))
                    high_val = float(np.max(vals))

                row[f"{metric}_center"] = center_val
                row[f"{metric}_low"] = low_val
                row[f"{metric}_high"] = high_val

            rows.append(row)

    if len(rows) == 0:
        raise ValueError("No center/interval rows were built.")

    return pd.DataFrame(rows)


# =========================================================
# Plotting
# =========================================================
def plot_metric_with_intervals(
    center_interval_df: pd.DataFrame,
    rare_type_df: pd.DataFrame,
    rare_threshold: float,
    metric_col: str,
    out_png: str,
    dpi: int = 300,
):
    center_col = f"{metric_col}_center"
    low_col = f"{metric_col}_low"
    high_col = f"{metric_col}_high"

    sub = center_interval_df.copy()
    if sub.empty or center_col not in sub.columns:
        logger.warning("Skipping plot for metric=%s", metric_col)
        return

    rare_sub = rare_type_df[rare_type_df["rare_threshold"] == rare_threshold].copy()
    rare_sub = rare_sub.sort_values(["n_cells", "cell_type"], ascending=[False, True])

    ordered_celltypes = [
        ct for ct in rare_sub["cell_type"].astype(str).tolist()
        if ct in set(sub["cell_type"]) and ct != "Class-switched B"
    ]
    methods_present = [m for m in METHOD_ORDER if m in set(sub["method"])]

    if len(ordered_celltypes) == 0 or len(methods_present) == 0:
        logger.warning("No ordered cell types or methods found for metric=%s", metric_col)
        return

    fig_w = max(10.0, 0.9 * len(ordered_celltypes))
    fig, ax = plt.subplots(figsize=(fig_w, 6.2))

    n_methods = len(methods_present)
    group_width = 0.8
    point_offsets = np.linspace(
        -group_width / 2 + group_width / (2 * n_methods),
        +group_width / 2 - group_width / (2 * n_methods),
        n_methods,
    )
    centers = np.arange(len(ordered_celltypes))

    for j, method in enumerate(methods_present):
        ss = sub[sub["method"] == method].copy()

        for i, ct in enumerate(ordered_celltypes):
            row = ss[ss["cell_type"].astype(str) == ct]
            if row.empty:
                continue

            row = row.iloc[0]
            x = centers[i] + point_offsets[j]
            y = float(row[center_col])

            low = row[low_col]
            high = row[high_col]

            if pd.notna(low) and pd.notna(high):
                ax.vlines(
                    x=x,
                    ymin=float(low),
                    ymax=float(high),
                    color=variant_to_color(method),
                    linewidth=3.5,
                    alpha=0.9,
                    zorder=2,
                )

            ax.plot(
                x,
                y,
                marker="o",
                linestyle="None",
                markersize=10,
                color=variant_to_color(method),
                zorder=3,
            )

    ax.set_xticks(centers)
    ax.set_xticklabels(ordered_celltypes, rotation=0, ha="center", fontsize=18)
    ax.set_ylabel(PRETTY_METRIC_LABELS.get(metric_col, metric_col), fontsize=24)

    for i in range(1, len(ordered_celltypes)):
        ax.axvline(i - 0.5, color="lightgray", linewidth=1.3, zorder=0)

    ax.tick_params(axis="x", width=2.0, length=6)
    ax.tick_params(axis="y", width=2.0, length=6, labelsize=20)
    ax.grid(False)

    fig.tight_layout()
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    fig.savefig(out_png.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved plot with intervals: %s", out_png)


# =========================================================
# Main
# =========================================================
def main():
    OUTPUT_ROOT = "/storage/home/dvl5760/work/camp_gr/rare_cell_pbmc_results"
    RARE_THRESHOLD = 0.01
    TARGET_N_METACELLS = 1000
    N_NEAREST = 3
    DPI = 300

    raw_csv = os.path.join(OUTPUT_ROOT, "PBMC_rare_cell_ALL_raw.csv")
    rare_type_csv = os.path.join(OUTPUT_ROOT, "PBMC_rare_cell_types.csv")

    out_root = os.path.join(
        OUTPUT_ROOT,
        f"plots_fixed_nmetacells_{TARGET_N_METACELLS}_rare{str(RARE_THRESHOLD).replace('.', 'p')}_core6_with_intervals"
    )
    plot_dir = os.path.join(out_root, "per_rare_cell_plots")
    safe_mkdir(out_root)
    safe_mkdir(plot_dir)

    if not os.path.exists(raw_csv):
        raise FileNotFoundError(f"Missing raw CSV: {raw_csv}")
    if not os.path.exists(rare_type_csv):
        raise FileNotFoundError(f"Missing rare type CSV: {rare_type_csv}")

    raw_all = pd.read_csv(raw_csv)
    rare_type_all = pd.read_csv(rare_type_csv)

    logger.info("Loaded raw CSV: %s", raw_csv)
    logger.info("Loaded rare type CSV: %s", rare_type_csv)

    selection_df, method_to_nearest_n = get_selected_resolution_table(
        raw_df=raw_all,
        target_n_metacells=TARGET_N_METACELLS,
        rare_threshold=RARE_THRESHOLD,
        n_nearest=N_NEAREST,
    )

    save_selection_table(
        selection_df=selection_df,
        out_csv=os.path.join(out_root, f"selected_center_rows_near_{TARGET_N_METACELLS}.csv"),
    )

    save_nearest_resolution_lists(
        selection_df=selection_df,
        method_to_nearest_n=method_to_nearest_n,
        out_csv=os.path.join(out_root, f"selected_{N_NEAREST}_nearest_resolutions_near_{TARGET_N_METACELLS}.csv"),
    )

    center_interval_df = build_center_and_interval_df(
        raw_df=raw_all,
        rare_threshold=RARE_THRESHOLD,
        selection_df=selection_df,
        method_to_nearest_n=method_to_nearest_n,
        metric_cols=CORE_METRICS_TO_PLOT,
    )

    center_interval_df.to_csv(
        os.path.join(out_root, "center_and_interval_values.csv"),
        index=False
    )
    logger.info("Saved center/interval table.")

    methods_present = [m for m in METHOD_ORDER if m in set(center_interval_df["method"])]
    save_shared_legend_png(
        out_png=os.path.join(out_root, "legend.png"),
        methods=methods_present,
        dpi=DPI,
    )

    save_rare_cell_order_txt(
        rare_type_df=rare_type_all,
        rare_threshold=RARE_THRESHOLD,
        out_txt=os.path.join(out_root, "rare_cell_order.txt"),
    )

    for metric in CORE_METRICS_TO_PLOT:
        out_png = os.path.join(
            plot_dir,
            f"PBMC_fixed_{TARGET_N_METACELLS}_{metric}_with_{N_NEAREST}nearest_interval.png"
        )
        plot_metric_with_intervals(
            center_interval_df=center_interval_df,
            rare_type_df=rare_type_all,
            rare_threshold=RARE_THRESHOLD,
            metric_col=metric,
            out_png=out_png,
            dpi=DPI,
        )

    logger.info(
        "Finished PBMC per-rare-cell plotting near %d metacells with %d-nearest intervals.",
        TARGET_N_METACELLS,
        N_NEAREST,
    )


if __name__ == "__main__":
    main()
