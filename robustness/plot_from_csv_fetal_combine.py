#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


plt.rcParams.update({
    "font.size": 24,          # base font
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
# Style / palette
# =========================================================
DISPLAY_NAME_MAP = {
    "camp1": "CAMP1",
    "camp2": "CAMP2",
    "camp3": "CAMP3",
    "camp4": "CAMP4",
    "seacells": "SEACells",
    "supercell": "SuperCell",
    "metacell": "MetaCell",
    "metacell2": "MetaCell2",
    "metaq": "MetaQ",
}

CUSTOM_PALETTE = {
    "CAMP1":     "#1f77b4",
    "CAMP2":     "#ff7f0e",
    "CAMP3":     "#2ca02c",
    "SEACells":  "#d62728",
    "SuperCell": "#9467bd",
    "MetaCell":  "#8c564b",
    "MetaCell2": "#e377c2",
    "MetaQ":     "#7f7f7f",
    "CAMP4":     "#bcbd22",
}

#GAMMA_TO_METACELLS = {
#    20: 1879,
#    30: 1253,
#    32: 1174,
#    35: 1074,
#    38: 989,
#    42: 895,
#    47: 800,
#    53: 709,
#    60: 626,
#    70: 537,
#}

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


VARIANT_ORDER = ["camp1", "camp2", "camp3", "camp4"]


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

def value_to_linestyle(variation_mode: str, value: int):
    if variation_mode == "hvg":
        return HVG_LINESTYLES.get(int(value), "-")
    return PCA_LINESTYLES.get(int(value), "-")


PRETTY_METRIC_LABELS = {
    "purity_mean": "Purity",
    "purity_median": "Purity",
    "entropy_mean": "Entropy",
    "entropy_median": "Entropy",
    "compactness_mean": "Compactness",
    "compactness_median": "Compactness",
    "separation_mean": "Separation",
    "separation_median": "Separation",
    "sc_ratio_mean": "SC ratio",
    "sc_ratio_median": "SC ratio",
    "INV_mean": "INV",
    "INV_median": "INV",
}


def variant_to_display(v: str) -> str:
    return DISPLAY_NAME_MAP.get(v, v)


def variant_to_color(v: str) -> str:
    return CUSTOM_PALETTE.get(variant_to_display(v), "#333333")


def safe_mkdir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def ecdf_xy(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.array([]), np.array([])
    x = np.sort(vals)
    y = np.arange(1, x.size + 1) / x.size
    return x, y

def get_gamma_closest_to_n_metacells(target_n_metacells: int) -> int:
    return min(
        GAMMA_TO_METACELLS.keys(),
        key=lambda g: abs(GAMMA_TO_METACELLS[g] - target_n_metacells)
    )

def asinh_transform(y, s):
    return np.arcsinh(y / s)

def asinh_inverse(yprime, s):
    return s * np.sinh(yprime)

def is_compactness_metric(metric_col: str) -> bool:
    return metric_col in {"compactness_mean", "compactness_median"}

def uses_scientific_notation(metric_col: str) -> bool:
    return metric_col in {
        "compactness_mean",
        "compactness_median",
        "separation_mean",
        "separation_median",
    }
# =========================================================
# Plotting helpers
# =========================================================
def get_gammas_near_target_n_metacells(target_n_metacells: int, n_neighbors: int = 3):
    gamma_dist = sorted(
        GAMMA_TO_METACELLS.keys(),
        key=lambda g: abs(GAMMA_TO_METACELLS[g] - target_n_metacells)
    )
    return gamma_dist[:n_neighbors]



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
                #linewidth=2.2,
                #markersize=6,
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
        #fontsize=12,
        fontsize=14,
        handlelength=2.0,
        columnspacing=1.4,
    )
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight", transparent=True)
    plt.close(fig)
    logger.info("Saved legend: %s", out_png)


def plot_metric_lines_single_variation(
    summary_df: pd.DataFrame,
    dataset_name: str,
    variation_mode: str,
    metric_col: str,
    variation_value: int,
    out_png: str,
    dpi: int = 600,
):
    sub = summary_df[summary_df["variation_value"] == variation_value].copy()
    if sub.empty or metric_col not in sub.columns:
        logger.warning("Skipping %s variation_value=%s", metric_col, variation_value)
        return

    sub = sub.sort_values(["gamma", "variant"])

    #fig, ax = plt.subplots(figsize=(6.2, 4.8))
    #fig, ax = plt.subplots(figsize=(7.5, 5.5))
    fig, ax = plt.subplots(figsize=(9, 6.5))

    variants_present = [v for v in VARIANT_ORDER if v in set(sub["variant"])]
    for variant in variants_present:
        #ss = sub[sub["variant"] == variant].sort_values("gamma")

        #x = ss["gamma"].values.astype(float)
        #x = ss["gamma"].map(GAMMA_TO_METACELLS).values.astype(float)
        
        ss = sub[sub["variant"] == variant].copy()
        ss["n_metacells"] = ss["gamma"].map(GAMMA_TO_METACELLS)
        ss = ss.sort_values("n_metacells")

        x = ss["n_metacells"].values.astype(float)


        y = ss[metric_col].values.astype(float)
        color = variant_to_color(variant)

        if metric_col.endswith("_mean"):
            base_metric = metric_col.replace("_mean", "")
        elif metric_col.endswith("_median"):
            base_metric = metric_col.replace("_median", "")
        else:
            base_metric = None

        if base_metric is not None:
            q25_col = f"{base_metric}_q25"
            q75_col = f"{base_metric}_q75"
            if q25_col in ss.columns and q75_col in ss.columns:
                y_q25 = ss[q25_col].values.astype(float)
                y_q75 = ss[q75_col].values.astype(float)
                #ax.fill_between(
                #    x,
                #    y_q25,
                #    y_q75,
                #    color=color,
                #    alpha=0.05,
                #    linewidth=0,
                #)

        ax.plot(
            x,
            y,
            marker="o",
            #linewidth=2.2,
            #markersize=5.5,
            linewidth=5.0,
            markersize=10,
            color=color,
        )

    #title_suffix = f"HVG={variation_value}" if variation_mode == "hvg" else f"PCs={variation_value}"
    #ax.set_title(f"{dataset_name} | {title_suffix}", fontsize=18)
    title_suffix = f"HVG={variation_value}" if variation_mode == "hvg" else f"PCs={variation_value}"
    ax.set_title(title_suffix, fontsize=28)
    
    #ax.set_xlabel("gamma", fontsize=16)
    #ax.set_xlabel("Number of Metacells", fontsize=16)
    ax.set_xlabel("")
    #ax.set_xticks(sorted(GAMMA_TO_METACELLS.values()))
    
    #ax.set_xticks([500, 1000, 1500, 2000])
    #ax.set_xlim(450, 2050)

    ax.set_xscale("log")
    ax.set_xticks([1e3, 1e4])
    ax.set_xticklabels([r"$10^3$", r"$10^4$"])
    ax.set_xlim(8e2, 3e4)

    #ax.set_ylabel(PRETTY_METRIC_LABELS.get(metric_col, metric_col), fontsize=16)
    ax.set_ylabel("")
    #ax.grid(True, alpha=0.25)
    ax.grid(False)
    ax.tick_params(labelsize=26, width=2.0, length=6)
    #ax.tick_params(labelsize=18, width=2.0, length=6)
    #ax.tick_params(labelsize=18, width=1.5)
    #ax.tick_params(labelsize=11)

    #fig.tight_layout()
    #fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    #plt.close(fig)
    #logger.info("Saved plot: %s", out_png)


    fig.tight_layout()

    # PNG for Overleaf editing
    fig.savefig(out_png, dpi=300, bbox_inches="tight")

    # PDF for final manuscript
    out_pdf = out_png.replace(".png", ".pdf")
    fig.savefig(out_pdf, bbox_inches="tight")

    plt.close(fig)
    logger.info("Saved plot: %s", out_png)
    logger.info("Saved plot: %s", out_pdf)


def plot_metric_overlay_all_values(
    summary_df: pd.DataFrame,
    dataset_name: str,
    variation_mode: str,
    metric_col: str,
    out_png: str,
    dpi: int = 600,
):
    if summary_df.empty or metric_col not in summary_df.columns:
        logger.warning("Skipping overlay plot for metric=%s", metric_col)
        return

    #use_asinh = is_compactness_metric(metric_col)
    use_asinh = False

    df_plot = summary_df.copy()
    df_plot["n_metacells"] = df_plot["gamma"].map(GAMMA_TO_METACELLS)
    df_plot = df_plot.dropna(subset=["n_metacells"])

    # Match RECOMB x-window logic
    #x_min, x_max = 300, 2300
    x_min, x_max = 8e2, 3e4
    dsub = df_plot.loc[df_plot["n_metacells"].between(x_min, x_max)].copy()

    s = None
    if use_asinh:
        y = dsub[metric_col].dropna().astype(float).to_numpy()
        if len(y) > 0:
            s = np.percentile(np.abs(y), 20)
            if s <= 0:
                s = max(np.max(np.abs(y)), 1e-12)

    fig, ax = plt.subplots(figsize=(9, 6.5))

    variants_present = [v for v in VARIANT_ORDER if v in set(dsub["variant"])]
    variation_values = sorted(dsub["variation_value"].unique())

    # faint individual curves
    for variant in variants_present:
        for vv in variation_values:
            ss = dsub[
                (dsub["variant"] == variant) &
                (dsub["variation_value"] == vv)
            ].copy()

            if ss.empty:
                continue

            ss = ss.sort_values("n_metacells")
            x = ss["n_metacells"].values.astype(float)
            y = ss[metric_col].values.astype(float)

            if use_asinh:
                y = asinh_transform(y, s)

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

    # thick mean curve across settings
    for variant in variants_present:
        ss = dsub[dsub["variant"] == variant].copy()
        if ss.empty:
            continue

        mean_df = (
            ss.groupby("n_metacells", as_index=False)[metric_col]
            .mean()
            .sort_values("n_metacells")
        )

        x_mean = mean_df["n_metacells"].values.astype(float)
        y_mean = mean_df[metric_col].values.astype(float)

        if use_asinh:
            y_mean = asinh_transform(y_mean, s)

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

    #ax.set_xticks([500, 1000, 1500, 2000])
    #ax.set_xlim(450, 2000)
    #ax.set_xlabel("Number of Metacells", fontsize=20)

    ax.set_xscale("log")
    ax.set_xticks([1e3, 1e4])
    ax.set_xticklabels([r"$10^3$", r"$10^4$"])
    ax.set_xlim(8e2, 3e4)

    if use_asinh:
        raw_vals = dsub[metric_col].dropna().astype(float).to_numpy()
        y_min, y_max = float(np.min(raw_vals)), float(np.max(raw_vals))

        tick_vals = np.array([0, -2e-6, -4e-6, -6e-6, -1e-5, -2e-5, -5e-5, -1e-4])
        tick_vals = tick_vals[(tick_vals >= y_min) & (tick_vals <= 0)]

        if tick_vals.size < 4:
            tick_vals = np.linspace(max(y_min, -1e-4), min(0.0, y_max), 7)

        ax.set_yticks(asinh_transform(tick_vals, s))
        ax.set_yticklabels([f"{tv * 1e5:.2f}" for tv in tick_vals])
        #ax.set_ylabel("Compactness (1e-5)", fontsize=24)
    else:
        pass
        #ax.set_ylabel(PRETTY_METRIC_LABELS.get(metric_col, metric_col), fontsize=24)
    ax.set_ylabel("")
    ax.grid(False)
    ax.tick_params(labelsize=28, width=2.0, length=6)
    #ax.yaxis.get_offset_text().set_fontsize(26)

    if uses_scientific_notation(metric_col):
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax.yaxis.get_offset_text().set_fontsize(26)


    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_png.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)

    logger.info("Saved overlay plot: %s", out_png)

def previous_plot_metric_overlay_all_values(
    summary_df: pd.DataFrame,
    dataset_name: str,
    variation_mode: str,
    metric_col: str,
    out_png: str,
    dpi: int = 600,
):
    if summary_df.empty or metric_col not in summary_df.columns:
        logger.warning("Skipping overlay plot for metric=%s", metric_col)
        return

    #use_asinh = is_compactness_metric(metric_col)
    use_asinh = False
    s = None
    if use_asinh:
        all_y = summary_df[metric_col].dropna().astype(float).to_numpy()
        if len(all_y) > 0:
            s = np.percentile(np.abs(all_y), 20)
            if s <= 0:
                s = max(np.max(np.abs(all_y)), 1e-12)

    fig, ax = plt.subplots(figsize=(9, 6.5))

    variants_present = [v for v in VARIANT_ORDER if v in set(summary_df["variant"])]
    variation_values = sorted(summary_df["variation_value"].unique())

    # 1) faint individual HVG/PCA curves
    for variant in variants_present:
        for vv in variation_values:
            ss = summary_df[
                (summary_df["variant"] == variant) &
                (summary_df["variation_value"] == vv)
            ].copy()

            if ss.empty:
                continue

            ss["n_metacells"] = ss["gamma"].map(GAMMA_TO_METACELLS)
            ss = ss.dropna(subset=["n_metacells"]).sort_values("n_metacells")

            x = ss["n_metacells"].values.astype(float)
            y = ss[metric_col].values.astype(float)

            if use_asinh:
                y = asinh_transform(y, s)

            ax.plot(
                x,
                y,
                color=variant_to_color(variant),
                linewidth=1.8,
                linestyle="-",
                marker=None,
                alpha=0.18,
                zorder=1,
            )

    # 2) thick mean-across-HVG/PCA curve per method
    for variant in variants_present:
        ss = summary_df[summary_df["variant"] == variant].copy()
        if ss.empty:
            continue

        ss["n_metacells"] = ss["gamma"].map(GAMMA_TO_METACELLS)
        ss = ss.dropna(subset=["n_metacells"])

        mean_df = (
            ss.groupby("n_metacells", as_index=False)[metric_col]
            .mean()
            .sort_values("n_metacells")
        )

        x_mean = mean_df["n_metacells"].values.astype(float)
        y_mean = mean_df[metric_col].values.astype(float)

        if use_asinh:
            y_mean = asinh_transform(y_mean, s)

        ax.plot(
            x_mean,
            y_mean,
            color=variant_to_color(variant),
            linewidth=4.8,
            linestyle="-",
            marker="o",
            markersize=7,
            alpha=1.0,
            zorder=3,
        )

    ax.set_xticks([500, 1000, 1500, 2000])
    ax.set_xlim(450, 2050)
    ax.set_xlabel("Number of Metacells", fontsize=20)

    if use_asinh:
        raw_vals = summary_df[metric_col].dropna().astype(float).to_numpy()
        y_min, y_max = float(np.min(raw_vals)), float(np.max(raw_vals))

        tick_vals = np.array([0, -2e-6, -4e-6, -6e-6, -1e-5, -2e-5, -5e-5, -1e-4])
        tick_vals = tick_vals[(tick_vals >= y_min) & (tick_vals <= max(y_max, 0))]

        if tick_vals.size < 4:
            tick_vals = np.linspace(y_min, y_max, 6)

        ax.set_yticks(asinh_transform(tick_vals, s))
        ax.set_yticklabels([f"{tv * 1e5:.2f}" for tv in tick_vals])
        ax.set_ylabel("Compactness (1e-5)", fontsize=20)
    else:
        ax.set_ylabel(PRETTY_METRIC_LABELS.get(metric_col, metric_col), fontsize=20)

    ax.grid(False)
    ax.tick_params(labelsize=22, width=2.0, length=6)

    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_png.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)

    logger.info("Saved overlay plot: %s", out_png)

#def previous_plot_metric_overlay_all_values(
#    summary_df: pd.DataFrame,
#    dataset_name: str,
#    variation_mode: str,
#    metric_col: str,
#    out_png: str,
#    dpi: int = 600,
#):
#    if summary_df.empty or metric_col not in summary_df.columns:
#        logger.warning("Skipping overlay plot for metric=%s", metric_col)
#        return
#
#    use_asinh = is_compactness_metric(metric_col)
#
#    s = None
#    if use_asinh:
#        all_y = summary_df[metric_col].dropna().astype(float).to_numpy()
#        if len(all_y) > 0:
#            s = np.percentile(np.abs(all_y), 20)
#            if s <= 0:
#                s = max(np.max(np.abs(all_y)), 1e-12)
#
#
#    fig, ax = plt.subplots(figsize=(9, 6.5))
#
#    variants_present = [v for v in VARIANT_ORDER if v in set(summary_df["variant"])]
#    variation_values = sorted(summary_df["variation_value"].unique())
#
#    # -------------------------------------------------
#    # 1) draw faint individual HVG/PCA curves
#    # -------------------------------------------------
#    for variant in variants_present:
#        ss = summary_df[summary_df["variant"] == variant].copy()
#        if ss.empty:
#            continue
#
#        ss["n_metacells"] = ss["gamma"].map(GAMMA_TO_METACELLS)
#        ss = ss.dropna(subset=["n_metacells"])
#
#        mean_df = (
#            ss.groupby("n_metacells", as_index=False)[metric_col]
#            .mean()
#            .sort_values("n_metacells")
#        )
#
#        x_mean = mean_df["n_metacells"].values.astype(float)
#        y_mean = mean_df[metric_col].values.astype(float)
#
#        if use_asinh:
#            y_mean = asinh_transform(y_mean, s)
#
#        ax.plot(
#            x_mean,
#            y_mean,
#            color=variant_to_color(variant),
#            linewidth=4.8,
#            linestyle="-",
#            marker="o",
#            markersize=7,
#            alpha=1.0,
#            zorder=3,
#        )
#    #for variant in variants_present:
#    #    for vv in variation_values:
#    #        ss = summary_df[
#    #            (summary_df["variant"] == variant) &
#    #            (summary_df["variation_value"] == vv)
#    #        ].copy()
#
#    #        if ss.empty:
#    #            continue
#
#    #        ss["n_metacells"] = ss["gamma"].map(GAMMA_TO_METACELLS)
#    #        ss = ss.dropna(subset=["n_metacells"]).sort_values("n_metacells")
#
#    #        x = ss["n_metacells"].values.astype(float)
#    #        #y = ss[metric_col].values.astype(float)
#
#    #        y = ss[metric_col].values.astype(float)
#    #        if use_asinh:
#    #            y = asinh_transform(y, s)
#
#    #        ax.plot(
#    #            x,
#    #            y,
#    #            color=variant_to_color(variant),
#    #            linewidth=1.8,
#    #            linestyle="-",
#    #            marker=None,
#    #            alpha=0.18,
#    #            zorder=1,
#    #        )
#
#    # -------------------------------------------------
#    # 2) draw thick mean-across-HVG/PCA curve per method
#    # -------------------------------------------------
#    for variant in variants_present:
#        ss = summary_df[summary_df["variant"] == variant].copy()
#        if ss.empty:
#            continue
#
#        ss["n_metacells"] = ss["gamma"].map(GAMMA_TO_METACELLS)
#        ss = ss.dropna(subset=["n_metacells"])
#
#        mean_df = (
#            ss.groupby("n_metacells", as_index=False)[metric_col]
#            .mean()
#            .sort_values("n_metacells")
#        )
#
#        ax.plot(
#            mean_df["n_metacells"].values.astype(float),
#            #mean_df[metric_col].values.astype(float),
#            y_mean = mean_df[metric_col].values.astype(float)
#            if use_asinh:
#                y_mean = asinh_transform(y_mean, s)
#
#            color=variant_to_color(variant),
#            linewidth=4.8,
#            linestyle="-",
#            marker="o",
#            markersize=7,
#            alpha=1.0,
#            zorder=3,
#        )
#
#    ax.set_xticks([500, 1000, 1500, 2000])
#    ax.set_xlim(450, 2050)
#    ax.set_xlabel("Number of Metacells", fontsize=20)
#    #ax.set_ylabel(PRETTY_METRIC_LABELS.get(metric_col, metric_col), fontsize=20)
#    if use_asinh:
#        raw_vals = summary_df[metric_col].dropna().astype(float).to_numpy()
#        y_min, y_max = float(np.min(raw_vals)), float(np.max(raw_vals))
#
#        tick_vals = np.array([0, -2e-6, -4e-6, -6e-6, -1e-5, -2e-5, -5e-5, -1e-4])
#        tick_vals = tick_vals[(tick_vals >= y_min) & (tick_vals <= max(y_max, 0))]
#
#        if tick_vals.size < 4:
#            tick_vals = np.linspace(y_min, y_max, 6)
#
#        ax.set_yticks(asinh_transform(tick_vals, s))
#        ax.set_yticklabels([f"{tv * 1e5:.2f}" for tv in tick_vals])
#        ax.set_ylabel("Compactness (1e-5)", fontsize=20)
#    else:
#        ax.set_ylabel(PRETTY_METRIC_LABELS.get(metric_col, metric_col), fontsize=20)
#
#    ax.grid(False)
#    ax.tick_params(labelsize=22, width=2.0, length=6)
#
#    fig.tight_layout()
#    fig.savefig(out_png, dpi=300, bbox_inches="tight")
#    fig.savefig(out_png.replace(".png", ".pdf"), bbox_inches="tight")
#    plt.close(fig)
#
#    logger.info("Saved overlay plot: %s", out_png)


def plot_metric_vs_variation_near_fixed_metacells(
    summary_df: pd.DataFrame,
    dataset_name: str,
    variation_mode: str,
    metric_col: str,
    target_n_metacells: int,
    out_png: str,
    dpi: int = 600,
    n_neighbor_gammas: int = 3,
):
    if summary_df.empty or metric_col not in summary_df.columns:
        logger.warning("Skipping near-fixed-metacell variation plot for metric=%s", metric_col)
        return

    chosen_gammas = get_gammas_near_target_n_metacells(
        target_n_metacells=target_n_metacells,
        n_neighbors=n_neighbor_gammas,
    )

    sub = summary_df[summary_df["gamma"].isin(chosen_gammas)].copy()
    if sub.empty:
        logger.warning("No rows found near target metacells=%s for metric=%s", target_n_metacells, metric_col)
        return

    #use_asinh = is_compactness_metric(metric_col)
    use_asinh = False

    s = None
    if use_asinh:
        all_y = sub[metric_col].dropna().astype(float).to_numpy()
        if len(all_y) > 0:
            s = np.percentile(np.abs(all_y), 20)
            if s <= 0:
                s = max(np.max(np.abs(all_y)), 1e-12)

    fig, ax = plt.subplots(figsize=(9, 6.5))
    variants_present = [v for v in VARIANT_ORDER if v in set(sub["variant"])]

    # faint lines for each nearby gamma
    for variant in variants_present:
        for gamma in chosen_gammas:
            ss = sub[(sub["variant"] == variant) & (sub["gamma"] == gamma)].copy()
            if ss.empty:
                continue

            ss = ss.sort_values("variation_value")
            x = ss["variation_value"].values.astype(float)
            y = ss[metric_col].values.astype(float)

            if use_asinh:
                y = asinh_transform(y, s)

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

    # thick mean across nearby gammas
    for variant in variants_present:
        ss = sub[sub["variant"] == variant].copy()
        if ss.empty:
            continue

        mean_df = (
            ss.groupby("variation_value", as_index=False)[metric_col]
            .mean()
            .sort_values("variation_value")
        )

        x_mean = mean_df["variation_value"].values.astype(float)
        y_mean = mean_df[metric_col].values.astype(float)

        if use_asinh:
            y_mean = asinh_transform(y_mean, s)

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

    #ax.set_xlabel("Number of HVGs" if variation_mode == "hvg" else "Number of PCs", fontsize=20)

    if use_asinh:
        raw_vals = sub[metric_col].dropna().astype(float).to_numpy()
        y_min, y_max = float(np.min(raw_vals)), float(np.max(raw_vals))

        tick_vals = np.array([0, -2e-6, -4e-6, -6e-6, -1e-5, -2e-5, -5e-5, -1e-4])
        tick_vals = tick_vals[(tick_vals >= y_min) & (tick_vals <= max(y_max, 0))]

        if tick_vals.size < 4:
            tick_vals = np.linspace(y_min, y_max, 6)

        ax.set_yticks(asinh_transform(tick_vals, s))
        ax.set_yticklabels([f"{tv * 1e5:.2f}" for tv in tick_vals])
        #ax.set_ylabel("Compactness (1e-5)", fontsize=24)
    else:
        pass
        #ax.set_ylabel(PRETTY_METRIC_LABELS.get(metric_col, metric_col), fontsize=24)
    ax.set_ylabel("")
    shown_metacells = [GAMMA_TO_METACELLS[g] for g in chosen_gammas]
    #ax.set_title(f"Metacells near {target_n_metacells}: {shown_metacells}", fontsize=24) 
    logger.info(shown_metacells)

    ax.grid(False)
    ax.tick_params(labelsize=28, width=2.0, length=6)
    #ax.yaxis.get_offset_text().set_fontsize(26)
    if uses_scientific_notation(metric_col):
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax.yaxis.get_offset_text().set_fontsize(26)


    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_png.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)

    logger.info("Saved near-fixed-metacell variation plot: %s", out_png)


def plot_metric_vs_variation_at_fixed_metacells(
    summary_df: pd.DataFrame,
    dataset_name: str,
    variation_mode: str,
    metric_col: str,
    target_n_metacells: int,
    out_png: str,
    dpi: int = 600,
):
    if summary_df.empty or metric_col not in summary_df.columns:
        logger.warning("Skipping fixed-metacell variation plot for metric=%s", metric_col)
        return

    chosen_gamma = get_gamma_closest_to_n_metacells(target_n_metacells)
    actual_n_metacells = GAMMA_TO_METACELLS[chosen_gamma]

    sub = summary_df[summary_df["gamma"] == chosen_gamma].copy()
    if sub.empty:
        logger.warning("No rows found for gamma=%s metric=%s", chosen_gamma, metric_col)
        return

    fig, ax = plt.subplots(figsize=(9, 6.5))


    #use_asinh = is_compactness_metric(metric_col)
    use_asinh = False

    s = None
    if use_asinh:
        all_y = sub[metric_col].dropna().astype(float).to_numpy()
        if len(all_y) > 0:
            s = np.percentile(np.abs(all_y), 20)
            if s <= 0:
                s = max(np.max(np.abs(all_y)), 1e-12)

    variants_present = [v for v in VARIANT_ORDER if v in set(sub["variant"])]

    for variant in variants_present:
        ss = sub[sub["variant"] == variant].copy()
        if ss.empty:
            continue

        ss = ss.sort_values("variation_value")

        x = ss["variation_value"].values.astype(float)
        #y = ss[metric_col].values.astype(float)

        y = ss[metric_col].values.astype(float)
        if use_asinh:
            y = asinh_transform(y, s)

        ax.plot(
            x,
            y,
            color=variant_to_color(variant),
            linewidth=4.8,
            linestyle="-",
            marker="o",
            markersize=7,
            alpha=1.0,
            zorder=3,
            label=variant_to_display(variant),
        )

    ax.set_xlabel("Number of HVGs" if variation_mode == "hvg" else "Number of PCs", fontsize=20)
    #ax.set_ylabel(PRETTY_METRIC_LABELS.get(metric_col, metric_col), fontsize=20)
    if use_asinh:
        raw_vals = sub[metric_col].dropna().astype(float).to_numpy()
        y_min, y_max = float(np.min(raw_vals)), float(np.max(raw_vals))

        tick_vals = np.array([0, -2e-6, -4e-6, -6e-6, -1e-5, -2e-5, -5e-5, -1e-4])
        tick_vals = tick_vals[(tick_vals >= y_min) & (tick_vals <= max(y_max, 0))]

        if tick_vals.size < 4:
            tick_vals = np.linspace(y_min, y_max, 6)

        ax.set_yticks(asinh_transform(tick_vals, s))
        ax.set_yticklabels([f"{tv * 1e5:.2f}" for tv in tick_vals])
        ax.set_ylabel("Compactness (1e-5)", fontsize=20)
    else:
        ax.set_ylabel(PRETTY_METRIC_LABELS.get(metric_col, metric_col), fontsize=20)

    ax.set_title(f"Metacells={actual_n_metacells}", fontsize=26)
    ax.grid(False)
    ax.tick_params(labelsize=22, width=2.0, length=6)

    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_png.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)

    logger.info("Saved fixed-metacell variation plot: %s", out_png)



#def previous_plot_metric_overlay_all_values(
#    summary_df: pd.DataFrame,
#    dataset_name: str,
#    variation_mode: str,
#    metric_col: str,
#    out_png: str,
#    dpi: int = 600,
#):
#    if summary_df.empty or metric_col not in summary_df.columns:
#        logger.warning("Skipping overlay plot for metric=%s", metric_col)
#        return
#
#    fig, ax = plt.subplots(figsize=(9, 6.5))
#
#    variants_present = [v for v in VARIANT_ORDER if v in set(summary_df["variant"])]
#    variation_values = sorted(summary_df["variation_value"].unique())
#
#    for variant in variants_present:
#        for vv in variation_values:
#            ss = summary_df[
#                (summary_df["variant"] == variant) &
#                (summary_df["variation_value"] == vv)
#            ].copy()
#
#            if ss.empty:
#                continue
#
#            ss["n_metacells"] = ss["gamma"].map(GAMMA_TO_METACELLS)
#            ss = ss.dropna(subset=["n_metacells"]).sort_values("n_metacells")
#
#            x = ss["n_metacells"].values.astype(float)
#            y = ss[metric_col].values.astype(float)
#
#            ax.plot(
#                x,
#                y,
#                color=variant_to_color(variant),
#                linestyle=value_to_linestyle(variation_mode, int(vv)),
#                linewidth=3.0,
#                marker="o",
#                markersize=5.0,
#                alpha=0.95,
#            )
#
#    ax.set_xticks([500, 1000, 1500, 2000])
#    ax.set_xlim(450, 2050)
#    ax.set_xlabel("Number of Metacells", fontsize=20)
#    ax.set_ylabel(PRETTY_METRIC_LABELS.get(metric_col, metric_col), fontsize=20)
#    ax.grid(False)
#    ax.tick_params(labelsize=22, width=2.0, length=6)
#
#    fig.tight_layout()
#    fig.savefig(out_png, dpi=300, bbox_inches="tight")
#    fig.savefig(out_png.replace(".png", ".pdf"), bbox_inches="tight")
#    plt.close(fig)
#
#    logger.info("Saved overlay plot: %s", out_png)


def save_overlay_style_legend_png(out_png: str, dpi: int = 600):
    handles = [
        Line2D(
            [0], [0],
            color="black",
            linewidth=1.8,
            alpha=0.25,
            label="Single PCA setting",
        ),
        Line2D(
            [0], [0],
            color="black",
            linewidth=4.8,
            marker="o",
            markersize=7,
            label="Mean across settings",
        ),
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
            alpha=0.3,
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

def plot_ecdf_sc_ratio_single_variation(
    raw_df: pd.DataFrame,
    dataset_name: str,
    variation_mode: str,
    variation_value: int,
    gamma: int,
    out_png: str,
    dpi: int = 600,
):
    sub = raw_df[
        (raw_df["variation_value"] == variation_value) &
        (raw_df["gamma"] == gamma)
    ].copy()

    if sub.empty or "sc_ratio" not in sub.columns:
        logger.warning("Skipping ECDF plot for variation_value=%s gamma=%s", variation_value, gamma)
        return

    fig, ax = plt.subplots(figsize=(6.2, 4.8))

    variants_present = [v for v in VARIANT_ORDER if v in set(sub["variant"])]
    for variant in variants_present:
        vals = sub.loc[sub["variant"] == variant, "sc_ratio"].values
        x, y = ecdf_xy(vals)
        if x.size == 0:
            continue
        ax.step(
            x,
            y,
            where="post",
            #linewidth=2.2,
            linewidth=5.0,
            color=variant_to_color(variant),
        )

    x_label_name = "HVG" if variation_mode == "hvg" else "PCs"
    #ax.set_title(
    #    n_metacells = GAMMA_TO_METACELLS.get(gamma, gamma)
    #    f"{dataset_name} | {x_label_name}={variation_value} | Metacells={n_metacells}",
    #    #f"{dataset_name} | {x_label_name}={variation_value} | gamma={gamma}",
    #    fontsize=18
    #)

    n_metacells = GAMMA_TO_METACELLS.get(gamma, gamma)
    ax.set_title(
        f"{x_label_name}={variation_value} | Metacells={n_metacells}",
        fontsize=28
    )

    #ax.set_xlabel("SC ratio", fontsize=16)
    #ax.set_ylabel("ECDF", fontsize=16)
    #ax.grid(True, alpha=0.25)
    ax.grid(False)
    ax.tick_params(labelsize=26, width=2.0, length=6)
    #ax.tick_params(labelsize=14, width=1.5)
    #ax.tick_params(labelsize=11)

    #fig.tight_layout()
    #fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    #plt.close(fig)
    #logger.info("Saved ECDF plot: %s", out_png)

    fig.tight_layout()

    # PNG for Overleaf editing
    fig.savefig(out_png, dpi=300, bbox_inches="tight")

    # PDF for final manuscript
    out_pdf = out_png.replace(".png", ".pdf")
    fig.savefig(out_pdf, bbox_inches="tight")

    plt.close(fig)
    logger.info("Saved ECDF plot: %s", out_png)
    logger.info("Saved ECDF plot: %s", out_pdf)


def make_all_plots_for_dataset_variation(
    summary_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    out_dir: str,
    dataset_name: str,
    variation_mode: str,
    dpi: int = 600,
    ecdf_gammas: Optional[List[int]] = None,
):
    metrics_to_plot = [
        "purity_mean",
        "entropy_mean",
        "compactness_mean",
        "separation_mean",
        "sc_ratio_median",
        "INV_mean",
    ]

    safe_mkdir(out_dir)

    legend_png = os.path.join(out_dir, f"{dataset_name}_{variation_mode}_legend.png")
    save_shared_legend_png(legend_png, variants=VARIANT_ORDER, dpi=dpi)

    gammas = sorted(summary_df["gamma"].unique())
    variation_values = sorted(summary_df["variation_value"].unique())


    save_method_legend_png(
        os.path.join(out_dir, f"{dataset_name}_{variation_mode}_method_legend.png"),
        variants=[v for v in VARIANT_ORDER if v in set(summary_df["variant"])],
        dpi=dpi,
    )

    save_value_legend_png(
        os.path.join(out_dir, f"{dataset_name}_{variation_mode}_value_legend.png"),
        variation_mode=variation_mode,
        values=variation_values,
        dpi=dpi,
    )


    save_overlay_style_legend_png(
        os.path.join(out_dir, f"{dataset_name}_{variation_mode}_overlay_style_legend.png"),
        dpi=dpi,
    )


    if ecdf_gammas is None:
        ecdf_gammas = [50, 200, 500]
    ecdf_gammas = [g for g in ecdf_gammas if g in gammas]

    # line plots: one PNG per metric per variation value
    #line_dir = os.path.join(out_dir, "line_plots")
    #safe_mkdir(line_dir)

    #for metric in metrics_to_plot:
    #    metric_dir = os.path.join(line_dir, metric)
    #    safe_mkdir(metric_dir)
    #    for vv in variation_values:
    #        out_png = os.path.join(
    #            metric_dir,
    #            f"{dataset_name}_{variation_mode}_{metric}_value{vv}.png"
    #        )
    #        plot_metric_lines_single_variation(
    #            summary_df=summary_df,
    #            dataset_name=dataset_name,
    #            variation_mode=variation_mode,
    #            metric_col=metric,
    #            variation_value=vv,
    #            out_png=out_png,
    #            dpi=dpi,
    #        )


    # reviewer-ready overlay plots: one PNG per metric, all HVG/PCA values together
    line_dir = os.path.join(out_dir, "overlay_metric_plots")
    safe_mkdir(line_dir)

    for metric in metrics_to_plot:
        out_png = os.path.join(
            line_dir,
            f"{dataset_name}_{variation_mode}_{metric}_overlay.png"
        )
        plot_metric_overlay_all_values(
            summary_df=summary_df,
            dataset_name=dataset_name,
            variation_mode=variation_mode,
            metric_col=metric,
            out_png=out_png,
            dpi=dpi,
        )

    fixed_nmetacell_dir = os.path.join(out_dir, "metric_vs_feature_near_fixed_metacells")
    safe_mkdir(fixed_nmetacell_dir)

    target_n_metacells = 3000

    save_near_fixed_metacell_style_legend_png(
        os.path.join(
            out_dir,
            f"{dataset_name}_{variation_mode}_near_fixed_metacell_style_legend.png"
        ),
        target_n_metacells=target_n_metacells,
        dpi=dpi,
    )


    for metric in metrics_to_plot:
        out_png = os.path.join(
            fixed_nmetacell_dir,
            f"{dataset_name}_{variation_mode}_{metric}_near_{target_n_metacells}_metacells.png"
        )
        plot_metric_vs_variation_near_fixed_metacells(
            summary_df=summary_df,
            dataset_name=dataset_name,
            variation_mode=variation_mode,
            metric_col=metric,
            target_n_metacells=target_n_metacells,
            out_png=out_png,
            dpi=dpi,
            n_neighbor_gammas=3,
        )


    #fixed_nmetacell_dir = os.path.join(out_dir, "metric_vs_feature_at_fixed_metacells")
    #safe_mkdir(fixed_nmetacell_dir)

    #target_n_metacells = 1000

    #for metric in metrics_to_plot:
    #    out_png = os.path.join(
    #        fixed_nmetacell_dir,
    #        f"{dataset_name}_{variation_mode}_{metric}_at_{target_n_metacells}_metacells.png"
    #    )
    #    plot_metric_vs_variation_at_fixed_metacells(
    #        summary_df=summary_df,
    #        dataset_name=dataset_name,
    #        variation_mode=variation_mode,
    #        metric_col=metric,
    #        target_n_metacells=target_n_metacells,
    #        out_png=out_png,
    #        dpi=dpi,
    #    )


    # ECDF plots of SC ratio
    ecdf_dir = os.path.join(out_dir, "ecdf_sc_ratio")
    safe_mkdir(ecdf_dir)

    for vv in variation_values:
        vv_dir = os.path.join(ecdf_dir, f"{'hvg' if variation_mode == 'hvg' else 'pcs'}_{vv}")
        safe_mkdir(vv_dir)

        for gamma in ecdf_gammas:
            out_png = os.path.join(
                vv_dir,
                f"{dataset_name}_{variation_mode}_ecdf_sc_ratio_value{vv}_gamma{gamma}.png"
            )
            plot_ecdf_sc_ratio_single_variation(
                raw_df=raw_df,
                dataset_name=dataset_name,
                variation_mode=variation_mode,
                variation_value=vv,
                gamma=gamma,
                out_png=out_png,
                dpi=dpi,
            )


# =========================================================
# Main
# =========================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Plot from existing summary/raw CSV files only.")

    parser.add_argument("--summary_csv", type=str, required=True)
    parser.add_argument("--raw_csv", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--variation_mode", type=str, choices=["hvg", "pca"], required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--ecdf_gammas", nargs="+", type=int, default=[50, 200, 500])

    return parser.parse_args()


def main():
    args = parse_args()
    logger.info("Arguments:\n%s", json.dumps(vars(args), indent=2))

    if not os.path.exists(args.summary_csv):
        raise FileNotFoundError(f"Missing summary CSV: {args.summary_csv}")
    if not os.path.exists(args.raw_csv):
        raise FileNotFoundError(f"Missing raw CSV: {args.raw_csv}")

    summary_df = pd.read_csv(args.summary_csv)
    raw_df = pd.read_csv(args.raw_csv)

    logger.info("Loaded summary_df shape: %s", summary_df.shape)
    logger.info("Loaded raw_df shape: %s", raw_df.shape)

    make_all_plots_for_dataset_variation(
        summary_df=summary_df,
        raw_df=raw_df,
        out_dir=args.out_dir,
        dataset_name=args.dataset_name,
        variation_mode=args.variation_mode,
        dpi=args.dpi,
        ecdf_gammas=args.ecdf_gammas,
    )

    logger.info("Plot-only regeneration finished.")


if __name__ == "__main__":
    main()
