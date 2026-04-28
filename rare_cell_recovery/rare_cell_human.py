#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("MKL_DYNAMIC", "FALSE")

from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


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


SHORT_CELLTYPE_MAP = {
    "Antigen presenting cells": "APC",
    "SATB2_LRRC7 positive cells": "SATB2_LRRC7",
    "PAEP_MECOM positive cells": "PAEP_MECOM",
    "Thymic epithelial cells": "Thymic epithelial",
    "Lens fibre cells": "Lens fibre",
    "CCL19_CCL21 positive cells": "CCL19_CCL21",
    "ELF3_AGBL2 positive cells": "ELF3_AGBL2",
    "AFP_ALB positive cells": "AFP_ALB",
    "CSH1_CSH2 positive cells": "CSH1_CSH2",
    "Corneal and conjunctival epithelial cells": "Corneal/Conj. epi.",
    "CLC_IL5RA positive cells": "CLC_IL5RA",
    "PDE1C_ACSM3 positive cells": "PDE1C_ACSM3",
    "SLC26A4_PAEP positive cells": "SLC26A4_PAEP",
}


#PRETTY_METRIC_LABELS = {
#    "dominant_capture_mean": "Dominant rare-cell capture",
#    "best_metacell_purity_mean": "Best rare-cell metacell purity",
#    "best_single_metacell_recall_mean": "Best single-metacell recall",
#    "capture_at_70_mean": "Rare-cell capture @ purity≥0.70",
#    "capture_at_90_mean": "Rare-cell capture @ purity≥0.90",
#}

PRETTY_METRIC_LABELS = {
    "bal_acc_rare": "Balanced Accuracy",

    "best_metacell_purity": "Best Metacell Purity",
    "mean_metacell_purity": "Mean Metacell Purity",
    "median_metacell_purity": "Median Metacell Purity",

    "best_single_metacell_recall": "Best Metacell Recall",
    "mean_metacell_recall": "Mean Metacell Recall",
    "median_metacell_recall": "Median Metacell Recall",

    "normalized_best_single_metacell_recall": "Balanced Best Metacell Recall",
    "normalized_mean_metacell_recall": "Balanced Mean Metacell Recall",
    "normalized_median_metacell_recall": "Balanced Median Metacell Recall",

    "dominant_capture": "Dominant Capture",
    "normalized_dominant_capture": "Balanced Dominant Capture",

    "capture_at_70": "Capture at Purity ≥ 0.70",
    "capture_at_90": "Capture at Purity ≥ 0.90",

    "bal_acc_rare": "Balanced Accuracy",
    "best_metacell_purity_mean": "Best Metacell Purity",
    "best_single_metacell_recall_mean": "Best Metacell Recall",
    "normalized_best_single_metacell_recall_mean": "Balanced Best Metacell Recall",
    "dominant_capture_mean": "Dominant Capture",
    "normalized_dominant_capture_mean": "Balanced Dominant Capture",
    "capture_at_70_mean": "Capture at Purity ≥ 0.70",
    "capture_at_90_mean": "Capture at Purity ≥ 0.90",

    "mean_metacell_purity_mean": "Mean Metacell Purity",
    "median_metacell_purity_mean": "Median Metacell Purity",
    "mean_metacell_recall_mean": "Mean Metacell Recall",
    "median_metacell_recall_mean": "Median Metacell Recall",
    "normalized_mean_metacell_recall_mean": "Balanced Mean Metacell Recall",
    "normalized_median_metacell_recall_mean": "Balanced Median Metacell Recall",

    "n_metacells_with_type": "Number of Metacells with Rare Type",
    "fragmentation_per_cell": "Fragmentation",
    "rare_entropy": "Entropy",
    "rare_entropy_normalized": "Balanced Entropy",
    "top2_recall_mass": "Top-2 Recall Mass",
    "top3_recall_mass": "Top-3 Recall Mass",
    #"best_f1": r"Best Metacell $F1^*$",
    "best_f1": r"Best Metacell $\mathrm{F}1^{*}$",

    "n_metacells_with_type_mean": "Mean Number of Metacells with Rare Type",
    "fragmentation_per_cell_mean": "Mean Fragmentation per Rare Cell",
    "rare_entropy_mean": "Mean Rare-cell Distribution Entropy",
    "rare_entropy_normalized_mean": "Mean Normalized Rare-cell Entropy",
    "top2_recall_mass_mean": "Mean Top-2 Recall Mass",
    "top3_recall_mass_mean": "Mean Top-3 Recall Mass",
    "best_f1_mean": "Mean Best Purity-Recall F1",

    "rare_recall": "Recall",
    "rare_precision": "Precision",
    "rare_f1": "F1",
    "rare_purity": "Purity",

    "rare_recall_mean": "Balanced Recall",
    "rare_precision_mean": "Balanced Precision",
    "rare_f1_mean": "Balanced F1",
    "rare_purity_mean": "Balanced Purity",

    "rare_ari": "Rare-cell",
    "rare_nmi": "Rare-cell",
}


def variant_to_display(v: str) -> str:
    return DISPLAY_NAME_MAP.get(v.lower(), v)


def variant_to_color(v: str) -> str:
    return CUSTOM_PALETTE.get(variant_to_display(v), "#333333")

def short_celltype_name(ct: str) -> str:
    return SHORT_CELLTYPE_MAP.get(ct, ct)


# =========================================================
# Utilities
# =========================================================
def safe_mkdir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def load_labels(input_h5ad: str, celltype_key: str) -> pd.Series:
    logger.info("Reading labels from %s", input_h5ad)
    adata = sc.read_h5ad(input_h5ad)
    if celltype_key not in adata.obs.columns:
        raise ValueError(f"celltype_key '{celltype_key}' not found in {input_h5ad}")
    labels = adata.obs[celltype_key].copy()
    labels.index = labels.index.astype(str)
    labels = labels.astype(str)
    logger.info("Loaded %d cells with %d unique labels", len(labels), labels.nunique())
    return labels


def load_partition_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, index_col=0)
    df.index = df.index.astype(str)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def normalize_partition_columns(df: pd.DataFrame) -> pd.DataFrame:
    new_cols = []
    for c in df.columns:
        s = str(c).strip()

        if s.startswith("X") and s[1:].replace(".", "", 1).isdigit():
            s = s[1:]

        try:
            f = float(s)
            if f.is_integer():
                s = str(int(f))
        except ValueError:
            pass

        new_cols.append(s)

    df = df.copy()
    df.columns = new_cols
    return df


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


# =========================================================
# Column mapping
# =========================================================

METHOD_CONFIG = [
    {
        "csv_name": "edit_4_partitions.csv",
        "method": "camp1",
        "column_map": {
            "100": {"gamma": 100, "n_metacells": 4945},
            "150": {"gamma": 150, "n_metacells": 3297},
            "200": {"gamma": 200, "n_metacells": 2472},
            "250": {"gamma": 250, "n_metacells": 1978},
            "300": {"gamma": 300, "n_metacells": 1648},
            "350": {"gamma": 350, "n_metacells": 1413},
            "400": {"gamma": 400, "n_metacells": 1236},
            "450": {"gamma": 450, "n_metacells": 1099},
            "500": {"gamma": 500, "n_metacells": 989},
            "550": {"gamma": 550, "n_metacells": 899},
        },
    },
    {
        "csv_name": "edit_4_partitions_large_metacell.csv",
        "method": "camp1",
        "column_map": {
            "20": {"gamma": 20, "n_metacells": 24729},
            "30": {"gamma": 30, "n_metacells": 16486},
            "40": {"gamma": 40, "n_metacells": 12364},
            "50": {"gamma": 50, "n_metacells": 9891},
            "60": {"gamma": 60, "n_metacells": 8243},
            "70": {"gamma": 70, "n_metacells": 7065},
        },
    },
    {
        "csv_name": "edit_4_add_simi_partitions.csv",
        "method": "camp2",
        "column_map": {
            "100": {"gamma": 100, "n_metacells": 4945},
            "150": {"gamma": 150, "n_metacells": 3297},
            "200": {"gamma": 200, "n_metacells": 2472},
            "250": {"gamma": 250, "n_metacells": 1978},
            "300": {"gamma": 300, "n_metacells": 1648},
            "350": {"gamma": 350, "n_metacells": 1413},
            "400": {"gamma": 400, "n_metacells": 1236},
            "450": {"gamma": 450, "n_metacells": 1099},
            "500": {"gamma": 500, "n_metacells": 989},
            "550": {"gamma": 550, "n_metacells": 899},
        },
    },
    {
        "csv_name": "edit_4_add_simi_partitions_large_metacell.csv",
        "method": "camp2",
        "column_map": {
            "20": {"gamma": 20, "n_metacells": 24729},
            "30": {"gamma": 30, "n_metacells": 16486},
            "40": {"gamma": 40, "n_metacells": 12364},
            "50": {"gamma": 50, "n_metacells": 9891},
            "60": {"gamma": 60, "n_metacells": 8243},
            "70": {"gamma": 70, "n_metacells": 7065},
        },
    },
    {
        "csv_name": "edit_4_add_ad_gau_partitions.csv",
        "method": "camp3",
        "column_map": {
            "100": {"gamma": 100, "n_metacells": 4945},
            "150": {"gamma": 150, "n_metacells": 3297},
            "200": {"gamma": 200, "n_metacells": 2472},
            "250": {"gamma": 250, "n_metacells": 1978},
            "300": {"gamma": 300, "n_metacells": 1648},
            "350": {"gamma": 350, "n_metacells": 1413},
            "400": {"gamma": 400, "n_metacells": 1236},
            "450": {"gamma": 450, "n_metacells": 1099},
            "500": {"gamma": 500, "n_metacells": 989},
            "550": {"gamma": 550, "n_metacells": 899},
        },
    },
    {
        "csv_name": "edit_4_add_ad_gau_partitions_large_metacell.csv",
        "method": "camp3",
        "column_map": {
            "20": {"gamma": 20, "n_metacells": 24729},
            "30": {"gamma": 30, "n_metacells": 16486},
            "40": {"gamma": 40, "n_metacells": 12364},
            "50": {"gamma": 50, "n_metacells": 9891},
            "60": {"gamma": 60, "n_metacells": 8243},
            "70": {"gamma": 70, "n_metacells": 7065},
        },
    },
    {
        "csv_name": "edit_5_partitions_full_metacell.csv",
        "method": "camp4",
        "column_map": {
            "100": {"gamma": 100, "n_metacells": 4945},
            "150": {"gamma": 150, "n_metacells": 3297},
            "200": {"gamma": 200, "n_metacells": 2472},
            "250": {"gamma": 250, "n_metacells": 1978},
            "300": {"gamma": 300, "n_metacells": 1648},
            "350": {"gamma": 350, "n_metacells": 1413},
            "400": {"gamma": 400, "n_metacells": 1236},
            "450": {"gamma": 450, "n_metacells": 1099},
            "500": {"gamma": 500, "n_metacells": 989},
            "550": {"gamma": 550, "n_metacells": 899},
            "20": {"gamma": 20, "n_metacells": 24729},
            "30": {"gamma": 30, "n_metacells": 16486},
            "40": {"gamma": 40, "n_metacells": 12364},
            "50": {"gamma": 50, "n_metacells": 9891},
            "60": {"gamma": 60, "n_metacells": 8243},
            "70": {"gamma": 70, "n_metacells": 7065},
        },
    },
    {
        "csv_name": "seacell_default_partition.csv",
        "method": "seacells",
        "column_map": {
            "100": {"gamma": 100, "n_metacells": 4945},
            "150": {"gamma": 150, "n_metacells": 3297},
            "200": {"gamma": 200, "n_metacells": 2472},
            "250": {"gamma": 250, "n_metacells": 1978},
            "300": {"gamma": 300, "n_metacells": 1648},
            "350": {"gamma": 350, "n_metacells": 1413},
            "400": {"gamma": 400, "n_metacells": 1236},
            "450": {"gamma": 450, "n_metacells": 1099},
            "500": {"gamma": 500, "n_metacells": 989},
            "550": {"gamma": 550, "n_metacells": 899},
            "20": {"gamma": 20, "n_metacells": 24729},
            "30": {"gamma": 30, "n_metacells": 16486},
            "40": {"gamma": 40, "n_metacells": 12364},
            "50": {"gamma": 50, "n_metacells": 9891},
            "60": {"gamma": 60, "n_metacells": 8243},
            "70": {"gamma": 70, "n_metacells": 7065},
        },
    },
    #{
    #    "csv_name": "supercell_membership_approx_new.csv",
    #    "method": "supercell",
    #    "column_map": {
    #        "100": {"gamma": 100, "n_metacells": 4945},
    #        "150": {"gamma": 150, "n_metacells": 3297},
    #        "200": {"gamma": 200, "n_metacells": 2472},
    #        "250": {"gamma": 250, "n_metacells": 1978},
    #        "300": {"gamma": 300, "n_metacells": 1648},
    #        "350": {"gamma": 350, "n_metacells": 1413},
    #        "400": {"gamma": 400, "n_metacells": 1236},
    #        "450": {"gamma": 450, "n_metacells": 1099},
    #        "500": {"gamma": 500, "n_metacells": 989},
    #        "550": {"gamma": 550, "n_metacells": 899},
    #    },
    #},
    
    {
        "csv_name": "supercell_membership_approx_new.csv",
        "method": "supercell",
        "column_map": {
            "4945": {"gamma": 100, "n_metacells": 4945},
            "3297": {"gamma": 150, "n_metacells": 3297},
            "2473": {"gamma": 200, "n_metacells": 2473},
            "1978": {"gamma": 250, "n_metacells": 1978},
            "1648": {"gamma": 300, "n_metacells": 1648},
            "1413": {"gamma": 350, "n_metacells": 1413},
            "1236": {"gamma": 400, "n_metacells": 1236},
            "1099": {"gamma": 450, "n_metacells": 1099},
            "989":  {"gamma": 500, "n_metacells": 989},
            "899":  {"gamma": 550, "n_metacells": 899},
        },
    },
    
    #{
    #    "csv_name": "supercell_membership_approx_new_large_metacell.csv",
    #    "method": "supercell",
    #    "column_map": {
    #        "20": {"gamma": 20, "n_metacells": 24729},
    #        "30": {"gamma": 30, "n_metacells": 16486},
    #        "40": {"gamma": 40, "n_metacells": 12364},
    #        "50": {"gamma": 50, "n_metacells": 9891},
    #        "60": {"gamma": 60, "n_metacells": 8243},
    #        "70": {"gamma": 70, "n_metacells": 7065},
    #    },
    #},

    {
        "csv_name": "supercell_membership_approx_new_large_metacell.csv",
        "method": "supercell",
        "column_map": {
            "24726": {"gamma": 20, "n_metacells": 24726},
            "16484": {"gamma": 30, "n_metacells": 16484},
            "12363": {"gamma": 40, "n_metacells": 12363},
            "9890":  {"gamma": 50, "n_metacells": 9890},
            "8242":  {"gamma": 60, "n_metacells": 8242},
            "7064":  {"gamma": 70, "n_metacells": 7064},
        },
    },

    #{
    #    "csv_name": "human_fetal_atlas_combined_metacell_labels.csv",
    #    "method": "metaq",
    #    "column_map": {
    #        "103": {"gamma": 103, "n_metacells": 4945},
    #        "152": {"gamma": 152, "n_metacells": 3297},
    #        "201": {"gamma": 201, "n_metacells": 2472},
    #        "250": {"gamma": 250, "n_metacells": 1978},
    #        "300": {"gamma": 300, "n_metacells": 1648},
    #        "350": {"gamma": 350, "n_metacells": 1413},
    #        "400": {"gamma": 400, "n_metacells": 1236},
    #        "450": {"gamma": 450, "n_metacells": 1099},
    #        "500": {"gamma": 500, "n_metacells": 989},
    #        "550": {"gamma": 550, "n_metacells": 899},
    #    },
    #},

    {
        "csv_name": "human_fetal_atlas_combined_metacell_labels.csv",
        "method": "metaq",
        "column_map": {
            "4945": {"gamma": 103, "n_metacells": 4945},
            "3297": {"gamma": 152, "n_metacells": 3297},
            "2472": {"gamma": 201, "n_metacells": 2472},
            "1978": {"gamma": 250, "n_metacells": 1978},
            "1648": {"gamma": 300, "n_metacells": 1648},
            "1413": {"gamma": 350, "n_metacells": 1413},
            "1236": {"gamma": 400, "n_metacells": 1236},
            "1099": {"gamma": 450, "n_metacells": 1099},
            "989":  {"gamma": 500, "n_metacells": 989},
            "899":  {"gamma": 550, "n_metacells": 899},
        },
    },


    {
        "csv_name": "human_fetal_atlas_combined_metacell_labels_large_metacell.csv",
        "method": "metaq",
        "column_map": {
            "24729": {"gamma": 20, "n_metacells": 24729},
            "16486": {"gamma": 30, "n_metacells": 16486},
            "12364": {"gamma": 40, "n_metacells": 12364},
            "9891":  {"gamma": 50, "n_metacells": 9891},
            "8243":  {"gamma": 60, "n_metacells": 8243},
            "7065":  {"gamma": 70, "n_metacells": 7065},
        },
    },

    #{
    #    "csv_name": "human_fetal_atlas_combined_metacell_labels_large_metacell.csv",
    #    "method": "metaq",
    #    "column_map": {
    #        "20": {"gamma": 20, "n_metacells": 24729},
    #        "30": {"gamma": 30, "n_metacells": 16486},
    #        "40": {"gamma": 40, "n_metacells": 12364},
    #        "50": {"gamma": 50, "n_metacells": 9891},
    #        "60": {"gamma": 60, "n_metacells": 8243},
    #        "70": {"gamma": 70, "n_metacells": 7065},
    #    },
    #},
]

#METHOD_CONFIG = [
#    {
#        "csv_name": "edit_4_seacell_covid_b_partitions.csv",
#        "method": "camp1",
#        "column_map": {
#            "70": {"gamma": 70, "n_metacells": 536},
#            "60": {"gamma": 60, "n_metacells": 626},
#            "53": {"gamma": 53, "n_metacells": 709},
#            "47": {"gamma": 47, "n_metacells": 799},
#            "42": {"gamma": 42, "n_metacells": 894},
#            "38": {"gamma": 38, "n_metacells": 989},
#            "35": {"gamma": 35, "n_metacells": 1073},
#            "32": {"gamma": 32, "n_metacells": 1174},
#            "30": {"gamma": 30, "n_metacells": 1252},
#            "20": {"gamma": 20, "n_metacells": 1879},
#        },
#    },
#    {
#        "csv_name": "edit_4_seacell_add_simi_covid_b_partitions.csv",
#        "method": "camp2",
#        "column_map": {
#            "70": {"gamma": 70, "n_metacells": 536},
#            "60": {"gamma": 60, "n_metacells": 626},
#            "53": {"gamma": 53, "n_metacells": 709},
#            "47": {"gamma": 47, "n_metacells": 799},
#            "42": {"gamma": 42, "n_metacells": 894},
#            "38": {"gamma": 38, "n_metacells": 989},
#            "35": {"gamma": 35, "n_metacells": 1073},
#            "32": {"gamma": 32, "n_metacells": 1174},
#            "30": {"gamma": 30, "n_metacells": 1252},
#            "20": {"gamma": 20, "n_metacells": 1879},
#        },
#    },
#    {
#        "csv_name": "edit_4_seacell_add_ad_gau_covid_b_partitions.csv",
#        "method": "camp3",
#        "column_map": {
#            "70": {"gamma": 70, "n_metacells": 536},
#            "60": {"gamma": 60, "n_metacells": 626},
#            "53": {"gamma": 53, "n_metacells": 709},
#            "47": {"gamma": 47, "n_metacells": 799},
#            "42": {"gamma": 42, "n_metacells": 894},
#            "38": {"gamma": 38, "n_metacells": 989},
#            "35": {"gamma": 35, "n_metacells": 1073},
#            "32": {"gamma": 32, "n_metacells": 1174},
#            "30": {"gamma": 30, "n_metacells": 1252},
#            "20": {"gamma": 20, "n_metacells": 1879},
#        },
#    },
#    {
#        "csv_name": "edit_5_partitions_full_metacell_covid_healthy.csv",
#        "method": "camp4",
#        "column_map": {
#            "70": {"gamma": 70, "n_metacells": 536},
#            "60": {"gamma": 60, "n_metacells": 626},
#            "53": {"gamma": 53, "n_metacells": 709},
#            "47": {"gamma": 47, "n_metacells": 799},
#            "42": {"gamma": 42, "n_metacells": 894},
#            "38": {"gamma": 38, "n_metacells": 989},
#            "35": {"gamma": 35, "n_metacells": 1073},
#            "32": {"gamma": 32, "n_metacells": 1174},
#            "30": {"gamma": 30, "n_metacells": 1252},
#            "20": {"gamma": 20, "n_metacells": 1879},
#        },
#    },
#    {
#        "csv_name": "seacell_default_partition.csv",
#        "method": "seacells",
#        "column_map": {
#            "70": {"gamma": 70, "n_metacells": 536},
#            "60": {"gamma": 60, "n_metacells": 626},
#            "53": {"gamma": 53, "n_metacells": 709},
#            "47": {"gamma": 47, "n_metacells": 799},
#            "42": {"gamma": 42, "n_metacells": 894},
#            "38": {"gamma": 38, "n_metacells": 989},
#            "35": {"gamma": 35, "n_metacells": 1073},
#            "32": {"gamma": 32, "n_metacells": 1174},
#            "30": {"gamma": 30, "n_metacells": 1252},
#            "20": {"gamma": 20, "n_metacells": 1879},
#        },
#    },
#    {
#        "csv_name": "supercell_membership.csv",
#        "method": "supercell",
#        "column_map": {
#            "736": {"gamma": 60, "n_metacells": 736},
#            "833": {"gamma": 53, "n_metacells": 833},
#            "940": {"gamma": 47, "n_metacells": 940},
#            "1052": {"gamma": 42, "n_metacells": 1052},
#            "1162": {"gamma": 38, "n_metacells": 1162},
#            "1262": {"gamma": 35, "n_metacells": 1262},
#            "1380": {"gamma": 32, "n_metacells": 1380},
#            "1472": {"gamma": 30, "n_metacells": 1472},
#            "2209": {"gamma": 20, "n_metacells": 2209},
#        },
#    },
#    {
#        "csv_name": "merged_metacell1.csv",
#        "method": "metacell1",
#        "column_map": {
#            "1": {"gamma": 25, "n_metacells": 1581},
#            "2": {"gamma": 33, "n_metacells": 1283},
#            "3": {"gamma": 38, "n_metacells": 1121},
#            "4": {"gamma": 44, "n_metacells": 996},
#            "5": {"gamma": 48, "n_metacells": 915},
#            "35": {"gamma": 19, "n_metacells": 1973},
#            "40": {"gamma": 21, "n_metacells": 1843},
#            "45": {"gamma": 23, "n_metacells": 1737},
#        },
#    },
#    {
#        "csv_name": "metacell2_membership_small_gamma.csv",
#        "method": "metacell2",
#        "column_map": {
#            "18": {"gamma": 18, "n_metacells": 638},
#            "16": {"gamma": 16, "n_metacells": 710},
#            "14": {"gamma": 14, "n_metacells": 800},
#            "12": {"gamma": 12, "n_metacells": 922},
#            "10": {"gamma": 10, "n_metacells": 1095},
#            "8": {"gamma": 8, "n_metacells": 1362},
#            "6": {"gamma": 6, "n_metacells": 1818},
#            "4": {"gamma": 4, "n_metacells": 2720},
#            "2": {"gamma": 2, "n_metacells": 6283},
#        },
#    },
#    {
#        "csv_name": "combined_metacell_labels.csv",
#        "method": "metaq",
#        "column_map": {
#            "500": {"gamma": 75, "n_metacells": 500},
#            "1000": {"gamma": 38, "n_metacells": 1000},
#            "1500": {"gamma": 25, "n_metacells": 1500},
#            "2000": {"gamma": 19, "n_metacells": 2000},
#        },
#    },
#]


# =========================================================
# Rare-cell metrics
# =========================================================
def compute_rare_types(cell_labels: pd.Series, rare_threshold: float) -> pd.DataFrame:
    counts = cell_labels.value_counts(dropna=False)
    freq = counts / counts.sum()
    rare = freq[freq <= rare_threshold].sort_values()

    return pd.DataFrame({
        "cell_type": rare.index.astype(str),
        "n_cells": counts.loc[rare.index].values.astype(int),
        "fraction": rare.values.astype(float),
    })


def compute_partition_bal_acc(labels: pd.Series, assign: pd.Series, rare_cell_types: List[str]) -> float:
    df = pd.DataFrame({
        "true_label": labels.astype(str).values,
        "metacell": assign.astype(str).values,
    }, index=labels.index)

    contingency = pd.crosstab(df["metacell"], df["true_label"])
    mc_majority_type = contingency.idxmax(axis=1)

    pred = df["metacell"].map(mc_majority_type).astype(str)
    true = df["true_label"].astype(str)

    rare_types = [ct for ct in rare_cell_types if ct in set(true)]
    if len(rare_types) == 0:
        return np.nan

    recalls = []
    for ct in rare_types:
        pos_mask = (true == ct)
        n_pos = int(pos_mask.sum())
        if n_pos == 0:
            continue
        tp = int(((pred == ct) & pos_mask).sum())
        recalls.append(tp / n_pos)

    if len(recalls) == 0:
        return np.nan
    return float(np.mean(recalls))


def compute_rare_partition_clustering_scores(
    labels: pd.Series,
    assign: pd.Series,
    rare_cell_types: List[str],
) -> Tuple[float, float]:
    mask = labels.astype(str).isin(set(rare_cell_types))
    if mask.sum() <= 1:
        return np.nan, np.nan

    y_true = labels.loc[mask].astype(str).values
    y_pred = assign.loc[mask].astype(str).values

    ari = adjusted_rand_score(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred)

    return float(ari), float(nmi)

def compute_rare_metrics_for_one_partition(
    labels: pd.Series,
    partition_df: pd.DataFrame,
    dataset_name: str,
    method_name: str,
    rare_threshold: float,
    column_map: Dict[str, Dict[str, int]],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    usable_cols = [str(c) for c in partition_df.columns if str(c) in column_map]
    usable_cols = sorted(usable_cols, key=lambda c: column_map[str(c)]["gamma"])

    if len(usable_cols) == 0:
        logger.error("No usable mapped columns found for method=%s", method_name)
        logger.error("Available columns: %s", list(partition_df.columns))
        logger.error("Expected mapped columns: %s", list(column_map.keys()))
        raise RuntimeError(f"No usable mapped columns found for method={method_name}")

    common_cells = labels.index.intersection(partition_df.index)
    if len(common_cells) == 0:
        raise RuntimeError(f"No overlapping cells for method={method_name}")

    labels_aligned = labels.loc[common_cells]
    part_aligned = partition_df.loc[common_cells, usable_cols].copy()

    rare_table = compute_rare_types(labels_aligned, rare_threshold)
    rare_cell_types = rare_table["cell_type"].astype(str).tolist()

    raw_rows = []
    summary_rows = []

    for col in usable_cols:
        col_str = str(col)
        gamma_int = int(column_map[col_str]["gamma"])
        n_metacells = int(column_map[col_str]["n_metacells"])

        assign = part_aligned[col].astype(str)
        df = pd.DataFrame({
            "cell_type": labels_aligned.values,
            "metacell": assign.values,
        }, index=common_cells)

        contingency = pd.crosstab(df["metacell"], df["cell_type"])
        mc_sizes = contingency.sum(axis=1)
        mc_majority_type = contingency.idxmax(axis=1)

        bal_acc = compute_partition_bal_acc(
            labels=labels_aligned,
            assign=assign,
            rare_cell_types=rare_cell_types,
        )

        rare_ari, rare_nmi = compute_rare_partition_clustering_scores(
            labels=labels_aligned,
            assign=assign,
            rare_cell_types=rare_cell_types,
        )

        per_type_rows = []

        for _, rr in rare_table.iterrows():
            ct = str(rr["cell_type"])
            n_ct = int(rr["n_cells"])
            frac_ct = float(rr["fraction"])

            if ct not in contingency.columns:
                continue

            ct_counts = contingency[ct]
            has_ct = ct_counts > 0

            majority_label_counts = {}
            if has_ct.sum() > 0:
                for mc in contingency.index[has_ct]:
                    maj = mc_majority_type.loc[mc]
                    majority_label_counts[maj] = majority_label_counts.get(maj, 0) + int(ct_counts.loc[mc])


            if has_ct.sum() == 0:
                best_purity = 0.0
                best_single_recall = 0.0
                normalized_best_single_recall = 0.0
                dominant_capture = 0.0
                normalized_dominant_capture = 0.0
                capture_at_70 = 0.0
                capture_at_90 = 0.0
                n_mcs_with_type = 0

                mean_metacell_purity = 0.0
                median_metacell_purity = 0.0
                mean_metacell_recall = 0.0
                median_metacell_recall = 0.0
                normalized_mean_metacell_recall = 0.0
                normalized_median_metacell_recall = 0.0

                fragmentation_per_cell = 0.0
                rare_entropy = 0.0
                rare_entropy_normalized = 0.0
                top2_recall_mass = 0.0
                top3_recall_mass = 0.0
                best_f1 = 0.0

                rare_recall = 0.0
                rare_precision = 0.0
                rare_f1 = 0.0
                rare_purity = 0.0

            else:
                purity_ct = ct_counts[has_ct] / mc_sizes[has_ct]
                recall_ct = ct_counts[has_ct] / n_ct
                majority_ct_mask = (mc_majority_type == ct)

                tp = float(ct_counts[majority_ct_mask].sum()) if majority_ct_mask.sum() > 0 else 0.0
                pred_pos = float(mc_sizes[majority_ct_mask].sum()) if majority_ct_mask.sum() > 0 else 0.0

                rare_recall = tp / n_ct if n_ct > 0 else 0.0
                rare_precision = tp / pred_pos if pred_pos > 0 else 0.0
                rare_purity = rare_precision

                if (rare_precision + rare_recall) > 0:
                    rare_f1 = 2.0 * rare_precision * rare_recall / (rare_precision + rare_recall)
                else:
                    rare_f1 = 0.0
                best_purity = float(purity_ct.max())
                best_single_recall = float(recall_ct.max())
                
                mean_metacell_purity = float(purity_ct.mean())
                median_metacell_purity = float(purity_ct.median())

                mean_metacell_recall = float(recall_ct.mean())
                median_metacell_recall = float(recall_ct.median())
                
                n_mcs_with_type = int(has_ct.sum())

                fragmentation_per_cell = float(n_mcs_with_type / n_ct) if n_ct > 0 else 0.0

                p = ct_counts[has_ct].values.astype(float)
                p = p / p.sum() if p.sum() > 0 else p

                rare_entropy = float(-(p * np.log(p + 1e-12)).sum()) if len(p) > 0 else 0.0
                if len(p) > 1:
                    rare_entropy_normalized = float(rare_entropy / np.log(len(p)))
                else:
                    rare_entropy_normalized = 0.0

                recall_vals_sorted = np.sort(recall_ct.values.astype(float))[::-1]
                top2_recall_mass = float(recall_vals_sorted[:2].sum()) if len(recall_vals_sorted) > 0 else 0.0
                top3_recall_mass = float(recall_vals_sorted[:3].sum()) if len(recall_vals_sorted) > 0 else 0.0

                if (best_purity + best_single_recall) > 0:
                    best_f1 = float(
                        2.0 * best_purity * best_single_recall / (best_purity + best_single_recall)
                    )
                else:
                    best_f1 = 0.0

                normalized_mean_metacell_recall = mean_metacell_recall / frac_ct if frac_ct > 0 else 0.0
                normalized_median_metacell_recall = median_metacell_recall / frac_ct if frac_ct > 0 else 0.0

                normalized_best_single_recall = best_single_recall / frac_ct if frac_ct > 0 else 0.0

                dominant_mask = has_ct & (mc_majority_type == ct)
                dominant_capture = float(ct_counts[dominant_mask].sum() / n_ct)
                normalized_dominant_capture = dominant_capture / frac_ct if frac_ct > 0 else 0.0

                capture70_mask = has_ct & ((ct_counts / mc_sizes) >= 0.70)
                capture90_mask = has_ct & ((ct_counts / mc_sizes) >= 0.90)

                capture_at_70 = float(ct_counts[capture70_mask].sum() / n_ct)
                capture_at_90 = float(ct_counts[capture90_mask].sum() / n_ct)
                #n_mcs_with_type = int(has_ct.sum())


            #if has_ct.sum() == 0:
            #    best_purity = 0.0
            #    best_single_recall = 0.0
            #    dominant_capture = 0.0
            #    capture_at_70 = 0.0
            #    capture_at_90 = 0.0
            #    n_mcs_with_type = 0
            #else:
            #    purity_ct = ct_counts[has_ct] / mc_sizes[has_ct]
            #    recall_ct = ct_counts[has_ct] / n_ct

            #    best_purity = float(purity_ct.max())
            #    best_single_recall = float(recall_ct.max())
                
            #    normalized_best_single_recall = best_single_recall / frac_ct if frac_ct > 0 else 0.0

            #    dominant_mask = has_ct & (mc_majority_type == ct)
            #    dominant_capture = float(ct_counts[dominant_mask].sum() / n_ct)
            #    normalized_dominant_capture = dominant_capture / frac_ct if frac_ct > 0 else 0.0

            #    capture70_mask = has_ct & ((ct_counts / mc_sizes) >= 0.70)
            #    capture90_mask = has_ct & ((ct_counts / mc_sizes) >= 0.90)

            #    capture_at_70 = float(ct_counts[capture70_mask].sum() / n_ct)
            #    capture_at_90 = float(ct_counts[capture90_mask].sum() / n_ct)
            #    n_mcs_with_type = int(has_ct.sum())

            per_type_rows.append({
                "dataset": dataset_name,
                "method": method_name,
                "gamma": gamma_int,
                "n_metacells": n_metacells,
                "rare_threshold": rare_threshold,
                "bal_acc_rare": bal_acc,
                "cell_type": ct,
                "n_cells_type": n_ct,
                "fraction_type": frac_ct,
                "n_metacells_with_type": n_mcs_with_type,
                "best_metacell_purity": best_purity,
                "best_single_metacell_recall": best_single_recall,
                "dominant_capture": dominant_capture,
                "capture_at_70": capture_at_70,
                "capture_at_90": capture_at_90,
                "top_majority_label": max(majority_label_counts, key=majority_label_counts.get) if len(majority_label_counts) > 0 else None,
                "top_majority_label_count": max(majority_label_counts.values()) if len(majority_label_counts) > 0 else 0,
                "normalized_best_single_metacell_recall": normalized_best_single_recall,
                "normalized_dominant_capture": normalized_dominant_capture,
                "mean_metacell_purity": mean_metacell_purity,
                "median_metacell_purity": median_metacell_purity,
                "mean_metacell_recall": mean_metacell_recall,
                "median_metacell_recall": median_metacell_recall,

                "normalized_mean_metacell_recall": normalized_mean_metacell_recall,
                "normalized_median_metacell_recall": normalized_median_metacell_recall,

                "fragmentation_per_cell": fragmentation_per_cell,
                "rare_entropy": rare_entropy,
                "rare_entropy_normalized": rare_entropy_normalized,
                "top2_recall_mass": top2_recall_mass,
                "top3_recall_mass": top3_recall_mass,
                "best_f1": best_f1,

                "rare_recall": rare_recall,
                "rare_precision": rare_precision,
                "rare_f1": rare_f1,
                "rare_purity": rare_purity,
            })

        raw_df = pd.DataFrame(per_type_rows)
        raw_rows.append(raw_df)

        if raw_df.empty:
            summary_rows.append({
                "dataset": dataset_name,
                "method": method_name,
                "gamma": gamma_int,
                "n_metacells": n_metacells,
                "rare_threshold": rare_threshold,
                "bal_acc_rare": bal_acc,
                "n_rare_types": 0,
                "best_metacell_purity_mean": np.nan,
                "best_single_metacell_recall_mean": np.nan,
                "dominant_capture_mean": np.nan,
                "capture_at_70_mean": np.nan,
                "capture_at_90_mean": np.nan,

                "mean_metacell_purity_mean": np.nan,
                "median_metacell_purity_mean": np.nan,
                "mean_metacell_recall_mean": np.nan,
                "median_metacell_recall_mean": np.nan,
                "normalized_mean_metacell_recall_mean": np.nan,
                "normalized_median_metacell_recall_mean": np.nan,

                "n_metacells_with_type_mean": np.nan,
                "fragmentation_per_cell_mean": np.nan,
                "rare_entropy_mean": np.nan,
                "rare_entropy_normalized_mean": np.nan,
                "top2_recall_mass_mean": np.nan,
                "top3_recall_mass_mean": np.nan,
                "best_f1_mean": np.nan,

                "rare_recall_mean": np.nan,
                "rare_precision_mean": np.nan,
                "rare_f1_mean": np.nan,
                "rare_purity_mean": np.nan,
                "rare_ari": np.nan,
                "rare_nmi": np.nan,
            })
        else:
            summary_rows.append({
                "dataset": dataset_name,
                "method": method_name,
                "gamma": gamma_int,
                "n_metacells": n_metacells,
                "rare_threshold": rare_threshold,
                "bal_acc_rare": bal_acc,
                "n_rare_types": int(raw_df["cell_type"].nunique()),
                "best_metacell_purity_mean": raw_df["best_metacell_purity"].mean(),
                "best_single_metacell_recall_mean": raw_df["best_single_metacell_recall"].mean(),
                "normalized_best_single_metacell_recall_mean": raw_df["normalized_best_single_metacell_recall"].mean(),
                "normalized_dominant_capture_mean": raw_df["normalized_dominant_capture"].mean(),
                "dominant_capture_mean": raw_df["dominant_capture"].mean(),
                "capture_at_70_mean": raw_df["capture_at_70"].mean(),
                "capture_at_90_mean": raw_df["capture_at_90"].mean(),

                "mean_metacell_purity_mean": raw_df["mean_metacell_purity"].mean(),
                "median_metacell_purity_mean": raw_df["median_metacell_purity"].mean(),
                "mean_metacell_recall_mean": raw_df["mean_metacell_recall"].mean(),
                "median_metacell_recall_mean": raw_df["median_metacell_recall"].mean(),

                "normalized_mean_metacell_recall_mean": raw_df["normalized_mean_metacell_recall"].mean(),
                "normalized_median_metacell_recall_mean": raw_df["normalized_median_metacell_recall"].mean(),

                "n_metacells_with_type_mean": raw_df["n_metacells_with_type"].mean(),
                "fragmentation_per_cell_mean": raw_df["fragmentation_per_cell"].mean(),
                "rare_entropy_mean": raw_df["rare_entropy"].mean(),
                "rare_entropy_normalized_mean": raw_df["rare_entropy_normalized"].mean(),
                "top2_recall_mass_mean": raw_df["top2_recall_mass"].mean(),
                "top3_recall_mass_mean": raw_df["top3_recall_mass"].mean(),
                "best_f1_mean": raw_df["best_f1"].mean(),

                "rare_recall_mean": raw_df["rare_recall"].mean(),
                "rare_precision_mean": raw_df["rare_precision"].mean(),
                "rare_f1_mean": raw_df["rare_f1"].mean(),
                "rare_purity_mean": raw_df["rare_purity"].mean(),
                "rare_ari": rare_ari,
                "rare_nmi": rare_nmi,
            })

    raw_all = pd.concat(raw_rows, ignore_index=True) if len(raw_rows) > 0 else pd.DataFrame()
    summary_all = pd.DataFrame(summary_rows)
    return raw_all, summary_all


# =========================================================
# Plotting
# =========================================================
def plot_metric(
    summary_df: pd.DataFrame,
    metric_col: str,
    rare_threshold: float,
    out_png: str,
    dpi: int = 600,
):
    sub = summary_df[summary_df["rare_threshold"] == rare_threshold].copy()
    
    #sub = sub[(sub["n_metacells"] >= 450) & (sub["n_metacells"] <= 2050)].copy()
    sub = sub[(sub["n_metacells"] >= 800) & (sub["n_metacells"] <= 30000)].copy()

    if sub.empty or metric_col not in sub.columns:
        logger.warning("Skipping plot for metric=%s rare_threshold=%.4f", metric_col, rare_threshold)
        return

    methods = [m for m in METHOD_ORDER if m in set(sub["method"])]

    #fig, ax = plt.subplots(figsize=(6.4, 4.8))
    fig, ax = plt.subplots(figsize=(7.2, 5.4))

    #x_max = float(sub["n_metacells"].max()) if len(sub) > 0 else 2000

    for method in methods:
        ss = sub[sub["method"] == method].sort_values("n_metacells")
        ax.plot(
            ss["n_metacells"].values.astype(float),
            ss[metric_col].values.astype(float),
            marker="o",
            linewidth=2.2,
            markersize=5.5,
            color=variant_to_color(method),
        )

    #ax.set_title(f"rare≤{rare_threshold:.3f}", fontsize=14)
    #ax.set_title(metric_col.replace("_mean",""), fontsize=22)
    #ax.set_xlabel("Number of Metacells", fontsize=12)
    ax.set_ylabel(PRETTY_METRIC_LABELS.get(metric_col, metric_col), fontsize=24)

    #ax.set_xticks([500, 1000, 1500, 2000])

    #right_lim = max(2050.0, x_max + 50.0)
    #ax.set_xlim(450, 2050)

    ax.set_xscale("log")
    ax.set_xticks([1e3, 1e4])
    ax.set_xticklabels([r"$10^3$", r"$10^4$"])
    ax.set_xlim(8e2, 3e4)

    ax.grid(False)
    #ax.tick_params(labelsize=11)
    ax.tick_params(labelsize=20, width=2.0, length=6)
    fig.tight_layout()
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    fig.savefig(out_png.replace(".png",".pdf"), bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved plot: %s", out_png)


def plot_per_method_rare_celltype_boxplots(
    raw_df: pd.DataFrame,
    rare_type_df: pd.DataFrame,
    metric_col: str,
    rare_threshold: float,
    method_name: str,
    out_png: str,
    dpi: int = 300,
):
    sub = raw_df[
        (raw_df["rare_threshold"] == rare_threshold) &
        (raw_df["method"] == method_name)
    ].copy()

    if sub.empty or metric_col not in sub.columns:
        logger.warning(
            "Skipping rare-celltype boxplot for method=%s metric=%s rare_threshold=%.4f",
            method_name, metric_col, rare_threshold
        )
        return

    rare_sub = rare_type_df[rare_type_df["rare_threshold"] == rare_threshold].copy()
    rare_sub = rare_sub.sort_values(["n_cells", "cell_type"], ascending=[False, True])

    ordered_celltypes = [ct for ct in rare_sub["cell_type"].astype(str).tolist() if ct in set(sub["cell_type"])]
    #ordered_celltypes = [
    #    ct for ct in rare_sub["cell_type"].astype(str).tolist()
    #    if ct in set(sub["cell_type"]) and ct != "Class-switched B"
    #]

    
    if len(ordered_celltypes) == 0:
        logger.warning("No ordered rare cell types found for method=%s", method_name)
        return

    data = []
    labels = []
    for ct in ordered_celltypes:
        vals = sub.loc[sub["cell_type"] == ct, metric_col].dropna().values.astype(float)
        if len(vals) > 0:
            data.append(vals)
            labels.append(ct)

    if len(data) == 0:
        logger.warning("No data for rare-celltype boxplot method=%s metric=%s", method_name, metric_col)
        return

    fig_w = max(8.0, 0.55 * len(labels))
    fig, ax = plt.subplots(figsize=(fig_w, 5.0))

    #bp = ax.boxplot(data, patch_artist=True, labels=labels, showfliers=False)
    bp = ax.boxplot(data, patch_artist=True, showfliers=False)

    #for patch in bp["boxes"]:
    #    patch.set_facecolor(variant_to_color(method))
    #    patch.set_alpha(0.85)
    #    patch.set_linewidth(1.6)

    for patch in bp["boxes"]:
        patch.set_facecolor(variant_to_color(method_name))
        patch.set_alpha(0.85)
        patch.set_linewidth(1.6)

    for element in ["whiskers", "caps"]:
        for item in bp[element]:
            item.set_linewidth(1.6)

    for item in bp["medians"]:
        item.set_linewidth(2.2)
    #for patch in bp["boxes"]:
    #    patch.set_facecolor(variant_to_color(method_name))
    #    patch.set_alpha(0.45)

    #ax.set_title(f"{variant_to_display(method_name)} | rare≤{rare_threshold:.3f}", fontsize=14)
    ax.set_title(variant_to_display(method_name), fontsize=18)
    #ax.set_xlabel("rare cell type", fontsize=12)
    #ax.set_ylabel(metric_col, fontsize=12)
    #ax.tick_params(axis="x", rotation=60, labelsize=9)
    #ax.tick_params(axis="y", labelsize=11)
    #ax.set_xticks(centers) 
    #ax.tick_params(axis="x", rotation=60, labelsize=16, width=2.0, length=6)
    #ax.set_xticks([])
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels([])

    ax.tick_params(axis="x", length=6, width=2.0)
    ax.tick_params(axis="y", labelsize=20, width=2.0, length=6)
    ax.grid(False)
    
    # save ordered rare-cell types (descending by abundance)
    #order_txt = out_png.replace(".png", "_rare_cell_order.txt")
    #with open(order_txt, "w") as f:
    #    for ct in labels:
    #        if ct != "class-switch B":
    #            f.write(f"{ct}\n")

    order_txt = out_png.replace(".png", "_rare_cell_order.txt")

    # get size lookup
    size_lookup = (
        rare_type_df
        .set_index("cell_type")["n_cells"]
        .to_dict()
    )

    with open(order_txt, "w") as f:
        for ct in ordered_celltypes:
            size = size_lookup.get(ct, "NA")
            f.write(f"{ct}\t{size}\n")

    fig.tight_layout()
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    fig.savefig(out_png.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved rare-celltype boxplot: %s", out_png)


def plot_grouped_celltype_method_boxplot(
    raw_df: pd.DataFrame,
    rare_type_df: pd.DataFrame,
    rare_threshold: float,
    metric_col: str,
    out_png: str,
    dpi: int = 300,
):
    sub = raw_df[raw_df["rare_threshold"] == rare_threshold].copy()

    if sub.empty or metric_col not in sub.columns:
        logger.warning(
            "Skipping grouped boxplot for metric=%s rare_threshold=%.4f",
            metric_col, rare_threshold
        )
        return

    rare_sub = rare_type_df[rare_type_df["rare_threshold"] == rare_threshold].copy()
    rare_sub = rare_sub.sort_values(["n_cells", "cell_type"], ascending=[False, True])

    ordered_celltypes = [
        ct for ct in rare_sub["cell_type"].astype(str).tolist()
        if ct in set(sub["cell_type"])
    ]
    methods_present = [m for m in METHOD_ORDER if m in set(sub["method"])]

    if len(ordered_celltypes) == 0 or len(methods_present) == 0:
        logger.warning("No ordered cell types or methods found for grouped boxplot.")
        return

    # -------- split into 3 rows --------
    row1_celltypes = ordered_celltypes[:4]
    row2_celltypes = ordered_celltypes[4:8]
    row3_celltypes = ordered_celltypes[8:]

    nrows = 3 if len(row3_celltypes) > 0 else (2 if len(row2_celltypes) > 0 else 1)

    # -------- compute one shared y-range for all panels --------
    yvals = pd.to_numeric(sub[metric_col], errors="coerce")
    yvals = yvals[np.isfinite(yvals)]

    if len(yvals) > 0:
        y_min = float(yvals.min())
        y_max = float(yvals.max())

        if y_max > y_min:
            pad = 0.05 * (y_max - y_min)
        else:
            pad = 0.05 * max(1.0, abs(y_max))

        shared_ylim = (y_min - pad, y_max + pad)
    else:
        shared_ylim = None

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=1,
        figsize=(15.5, 13.0 if nrows == 3 else 10.5),
        squeeze=False,
        sharey=True,
    )
    axes = axes.flatten()

    def draw_one_axis(ax, celltypes_for_row):
        n_methods = len(methods_present)
        group_width = 0.8
        box_width = group_width / max(n_methods, 1)

        centers = np.arange(len(celltypes_for_row))

        for j, method in enumerate(methods_present):
            method_data = []
            positions = []

            for i, ct in enumerate(celltypes_for_row):
                vals = sub.loc[
                    (sub["cell_type"] == ct) & (sub["method"] == method),
                    metric_col
                ].dropna().values.astype(float)

                if len(vals) == 0:
                    vals = np.array([np.nan])

                method_data.append(vals)
                pos = centers[i] - group_width / 2 + (j + 0.5) * box_width
                positions.append(pos)

            bp = ax.boxplot(
                method_data,
                positions=positions,
                widths=box_width * 1.05,
                patch_artist=True,
                showfliers=False,
                manage_ticks=False,
            )

            for patch in bp["boxes"]:
                patch.set_facecolor(variant_to_color(method))
                patch.set_alpha(0.85)
                patch.set_linewidth(1.6)

            for element in ["whiskers", "caps"]:
                for item in bp[element]:
                    item.set_color(variant_to_color(method))
                    item.set_linewidth(1.6)

            for item in bp["medians"]:
                item.set_color(variant_to_color(method))
                item.set_linewidth(2.2)

        for i in range(1, len(celltypes_for_row)):
            ax.axvline(i - 0.5, color="lightgray", linewidth=1.3, zorder=0)

        ax.set_xticks(centers)
        ax.set_xticklabels(
            [short_celltype_name(ct) for ct in celltypes_for_row],
            rotation=0,
            ha="center",
            fontsize=25,
        )
        ax.tick_params(axis="x", width=2.0, length=6)
        ax.tick_params(axis="y", width=2.0, length=6, labelsize=20)
        ax.grid(False)

        # force identical y-axis range across all rows
        if shared_ylim is not None:
            ax.set_ylim(shared_ylim)

    # row 1
    draw_one_axis(axes[0], row1_celltypes)
    axes[0].set_ylabel(PRETTY_METRIC_LABELS.get(metric_col, metric_col), fontsize=26)

    # row 2
    if len(row2_celltypes) > 0:
        draw_one_axis(axes[1], row2_celltypes)
        axes[1].set_ylabel(PRETTY_METRIC_LABELS.get(metric_col, metric_col), fontsize=26)

    # row 3
    if len(row3_celltypes) > 0:
        draw_one_axis(axes[2], row3_celltypes)
        axes[2].set_ylabel(PRETTY_METRIC_LABELS.get(metric_col, metric_col), fontsize=26)

    fig.tight_layout()
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    fig.savefig(out_png.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved grouped celltype-method boxplot (shared y-axis across rows): %s", out_png)


def different_y_axis_plot_grouped_celltype_method_boxplot(
    raw_df: pd.DataFrame,
    rare_type_df: pd.DataFrame,
    rare_threshold: float,
    metric_col: str,
    out_png: str,
    dpi: int = 300,
):
    sub = raw_df[raw_df["rare_threshold"] == rare_threshold].copy()

    if sub.empty or metric_col not in sub.columns:
        logger.warning(
            "Skipping grouped boxplot for metric=%s rare_threshold=%.4f",
            metric_col, rare_threshold
        )
        return

    rare_sub = rare_type_df[rare_type_df["rare_threshold"] == rare_threshold].copy()
    rare_sub = rare_sub.sort_values(["n_cells", "cell_type"], ascending=[False, True])

    ordered_celltypes = [
        ct for ct in rare_sub["cell_type"].astype(str).tolist()
        if ct in set(sub["cell_type"])
    ]
    methods_present = [m for m in METHOD_ORDER if m in set(sub["method"])]

    if len(ordered_celltypes) == 0 or len(methods_present) == 0:
        logger.warning("No ordered cell types or methods found for grouped boxplot.")
        return

    # -------- split into 2 rows: first 7, remaining --------
    #split_idx = 7
    #row1_celltypes = ordered_celltypes[:split_idx]
    #row2_celltypes = ordered_celltypes[split_idx:]

    row1_celltypes = ordered_celltypes[:4]
    row2_celltypes = ordered_celltypes[4:8]
    row3_celltypes = ordered_celltypes[8:]

    nrows = 3 if len(row3_celltypes) > 0 else (2 if len(row2_celltypes) > 0 else 1)

    #nrows = 2 if len(row2_celltypes) > 0 else 1
    #fig, axes = plt.subplots(
    #    nrows=nrows,
    #    ncols=1,
    #    figsize=(max(14.0, 1.25 * max(len(row1_celltypes), len(row2_celltypes) if len(row2_celltypes) > 0 else 0)), 10.5),
    #    squeeze=False,
    #)
    #axes = axes.flatten()


    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=1,
        figsize=(15.5, 13.0 if nrows == 3 else 10.5),
        squeeze=False,
    )
    axes = axes.flatten()

    legend_handles = []

    def draw_one_axis(ax, celltypes_for_row):
        n_methods = len(methods_present)
        group_width = 0.8
        box_width = group_width / max(n_methods, 1)

        centers = np.arange(len(celltypes_for_row))

        for j, method in enumerate(methods_present):
            method_data = []
            positions = []

            for i, ct in enumerate(celltypes_for_row):
                vals = sub.loc[
                    (sub["cell_type"] == ct) & (sub["method"] == method),
                    metric_col
                ].dropna().values.astype(float)

                if len(vals) == 0:
                    vals = np.array([np.nan])

                method_data.append(vals)
                pos = centers[i] - group_width / 2 + (j + 0.5) * box_width
                positions.append(pos)

            bp = ax.boxplot(
                method_data,
                positions=positions,
                widths=box_width * 1.05,
                patch_artist=True,
                showfliers=False,
                manage_ticks=False,
            )

            for patch in bp["boxes"]:
                patch.set_facecolor(variant_to_color(method))
                patch.set_alpha(0.85)
                patch.set_linewidth(1.6)

            for element in ["whiskers", "caps"]:
                for item in bp[element]:
                    item.set_color(variant_to_color(method))
                    item.set_linewidth(1.6)

            for item in bp["medians"]:
                item.set_color(variant_to_color(method))
                item.set_linewidth(2.2)

        for i in range(1, len(celltypes_for_row)):
            ax.axvline(i - 0.5, color="lightgray", linewidth=1.3, zorder=0)

        ax.set_xticks(centers)
        #ax.set_xticklabels(celltypes_for_row, rotation=0, ha="center", fontsize=8)
        ax.set_xticklabels(
            [short_celltype_name(ct) for ct in celltypes_for_row],
            rotation=0,
            ha="center",
            fontsize=25,
        )
        ax.tick_params(axis="x", width=2.0, length=6)
        ax.tick_params(axis="y", width=2.0, length=6, labelsize=20)
        ax.grid(False)

    # draw row 1
    draw_one_axis(axes[0], row1_celltypes)
    axes[0].set_ylabel(PRETTY_METRIC_LABELS.get(metric_col, metric_col), fontsize=26)

    # draw row 2 if needed
    if len(row2_celltypes) > 0:
        draw_one_axis(axes[1], row2_celltypes)
        axes[1].set_ylabel(PRETTY_METRIC_LABELS.get(metric_col, metric_col), fontsize=26)


    if len(row3_celltypes) > 0:
        draw_one_axis(axes[2], row3_celltypes)
        axes[2].set_ylabel(PRETTY_METRIC_LABELS.get(metric_col, metric_col), fontsize=26)

    # legend once for whole figure
    #for method in methods_present:
    #    legend_handles.append(
    #        Line2D(
    #            [0], [0],
    #            color=variant_to_color(method),
    #            lw=6,
    #            label=variant_to_display(method),
    #        )
    #    )

    #fig.legend(
    #    handles=legend_handles,
    #    loc="upper center",
    #    bbox_to_anchor=(0.5, 1.01),
    #    ncol=min(len(methods_present), 9),
    #    frameon=False,
    #    fontsize=14,
    #)

    #fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.tight_layout()
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    fig.savefig(out_png.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved grouped celltype-method boxplot (2-row layout): %s", out_png)


def one_panel_plot_grouped_celltype_method_boxplot(
    raw_df: pd.DataFrame,
    rare_type_df: pd.DataFrame,
    rare_threshold: float,
    metric_col: str,
    out_png: str,
    dpi: int = 300,
):
    sub = raw_df[raw_df["rare_threshold"] == rare_threshold].copy()

    if sub.empty or metric_col not in sub.columns:
        logger.warning(
            "Skipping grouped boxplot for metric=%s rare_threshold=%.4f",
            metric_col, rare_threshold
        )
        return

    rare_sub = rare_type_df[rare_type_df["rare_threshold"] == rare_threshold].copy()
    rare_sub = rare_sub.sort_values(["n_cells", "cell_type"], ascending=[False, True])

    ordered_celltypes = [
        ct for ct in rare_sub["cell_type"].astype(str).tolist()
        if ct in set(sub["cell_type"]) and ct != "Class-switched B"
    ]
    methods_present = [m for m in METHOD_ORDER if m in set(sub["method"])]

    if len(ordered_celltypes) == 0 or len(methods_present) == 0:
        logger.warning("No ordered cell types or methods found for grouped boxplot.")
        return

    fig_w = max(10.0, 0.9 * len(ordered_celltypes))
    fig, ax = plt.subplots(figsize=(fig_w, 6.2))

    n_methods = len(methods_present)
    group_width = 0.8
    box_width = group_width / max(n_methods, 1)

    centers = np.arange(len(ordered_celltypes))
    legend_handles = []

    for j, method in enumerate(methods_present):
        method_data = []
        positions = []

        for i, ct in enumerate(ordered_celltypes):
            vals = sub.loc[
                (sub["cell_type"] == ct) & (sub["method"] == method),
                metric_col
            ].dropna().values.astype(float)

            if len(vals) == 0:
                vals = np.array([np.nan])

            method_data.append(vals)

            pos = centers[i] - group_width / 2 + (j + 0.5) * box_width
            positions.append(pos)

        bp = ax.boxplot(
            method_data,
            positions=positions,
            widths=box_width * 1.05,
            patch_artist=True,
            showfliers=False,
            manage_ticks=False,
        )

        #for patch in bp["boxes"]:
        #    patch.set_facecolor(variant_to_color(method))
        #    patch.set_alpha(0.45)

        #for element in ["whiskers", "caps", "medians"]:
        #    for item in bp[element]:
        #        item.set_color(variant_to_color(method))

        for patch in bp["boxes"]:
            patch.set_facecolor(variant_to_color(method))
            patch.set_alpha(0.85)
            patch.set_linewidth(1.6)

        for element in ["whiskers", "caps"]:
            for item in bp[element]:
                item.set_color(variant_to_color(method))
                item.set_linewidth(1.6)

        for item in bp["medians"]:
            item.set_color(variant_to_color(method))
            item.set_linewidth(2.2)


        legend_handles.append(
            Line2D(
                [0], [0],
                color=variant_to_color(method),
                lw=6,
                label=variant_to_display(method),
            )
        )

    ax.set_xticks(centers)
    ax.set_xticklabels([])
    for i in range(1, len(ordered_celltypes)):
        ax.axvline(i - 0.5, color="lightgray", linewidth=1.3, zorder=0)
    
    #ax.set_xticklabels(ordered_celltypes, rotation=60, ha="right")
    #ax.set_xticklabels(ordered_celltypes, rotation=0, ha="center", fontsize=18)
    ax.set_xticklabels([])
    ax.set_ylabel(PRETTY_METRIC_LABELS.get(metric_col, metric_col), fontsize=24)
    #ax.set_title(metric_col.replace("_mean", "").replace("_", " "), fontsize=18)
    ax.tick_params(axis="x", width=2.0, length=6)
    ax.tick_params(axis="y", width=2.0, length=6, labelsize=20)
    ax.grid(False)

    #ax.legend(
    #    handles=legend_handles,
    #    loc="upper center",
    #    bbox_to_anchor=(0.5, 1.20),
    #    ncol=min(len(methods_present), 5),
    #    frameon=False,
    #    fontsize=12,
    #)

    fig.tight_layout()
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    fig.savefig(out_png.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved grouped celltype-method boxplot: %s", out_png)


def previous_plot_combined_method_boxplot(
    raw_df: pd.DataFrame,
    rare_threshold: float,
    metric_col: str,
    out_png: str,
    dpi: int = 300,
):
    sub = raw_df[raw_df["rare_threshold"] == rare_threshold].copy()

    if sub.empty or metric_col not in sub.columns:
        logger.warning(
            "Skipping combined method boxplot for metric=%s rare_threshold=%.4f",
            metric_col, rare_threshold
        )
        return

    methods_present = [m for m in METHOD_ORDER if m in set(sub["method"])]
    data = []
    labels = []

    for method in methods_present:
        vals = sub.loc[sub["method"] == method, metric_col].dropna().values.astype(float)
        if len(vals) > 0:
            data.append(vals)
            labels.append(method)

    if len(data) == 0:
        logger.warning("No data for combined method boxplot metric=%s", metric_col)
        return

    fig_w = max(8.0, 0.8 * len(labels))
    fig, ax = plt.subplots(figsize=(fig_w, 5.8))

    bp = ax.boxplot(data, patch_artist=True, showfliers=False)

    for patch, method in zip(bp["boxes"], labels):
        patch.set_facecolor(variant_to_color(method))
        patch.set_alpha(0.45)

    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels([variant_to_display(m) for m in labels], rotation=45, ha="right")

    ax.set_title(metric_col.replace("_mean", "").replace("_", " "), fontsize=18)
    ax.set_ylabel(PRETTY_METRIC_LABELS.get(metric_col, metric_col), fontsize=18)
    ax.tick_params(axis="x", labelsize=20, width=2.0, length=6)
    ax.tick_params(axis="y", labelsize=20, width=2.0, length=6)
    ax.grid(False)

    fig.tight_layout()
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    fig.savefig(out_png.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved combined method boxplot: %s", out_png)


def plot_per_method_assignment_heatmap(
    raw_df: pd.DataFrame,
    rare_type_df: pd.DataFrame,
    rare_threshold: float,
    method_name: str,
    out_png: str,
    dpi: int = 300,
):
    sub = raw_df[
        (raw_df["rare_threshold"] == rare_threshold) &
        (raw_df["method"] == method_name)
    ].copy()

    if sub.empty or "top_majority_label" not in sub.columns:
        logger.warning(
            "Skipping assignment heatmap for method=%s rare_threshold=%.4f",
            method_name, rare_threshold
        )
        return

    rare_sub = rare_type_df[rare_type_df["rare_threshold"] == rare_threshold].copy()
    rare_sub = rare_sub.sort_values(["n_cells", "cell_type"], ascending=[False, True])
    ordered_rare_types = rare_sub["cell_type"].astype(str).tolist()

    agg = (
        sub.groupby(["cell_type", "top_majority_label"], as_index=False)["top_majority_label_count"]
        .mean()
    )

    heat = agg.pivot(
        index="cell_type",
        columns="top_majority_label",
        values="top_majority_label_count"
    ).fillna(0.0)

    ordered_rare_types = [ct for ct in ordered_rare_types if ct in heat.index]
    heat = heat.loc[ordered_rare_types]

    same_axis_labels = [ct for ct in ordered_rare_types if ct in heat.columns]
    if len(same_axis_labels) > 0:
        heat = heat.reindex(columns=same_axis_labels, fill_value=0.0)
    else:
        col_order = heat.sum(axis=0).sort_values(ascending=False).index.tolist()
        heat = heat[col_order]

    bal_acc_val = sub["bal_acc_rare"].dropna()
    bal_acc_text = f"{bal_acc_val.mean():.3f}" if len(bal_acc_val) > 0 else "NA"

    fig_h = max(4.8, 0.45 * len(heat.index))
    fig_w = max(6.5, 0.45 * len(heat.columns))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(heat.values, aspect="auto")

    ax.set_xticks(np.arange(len(heat.columns)))
    #ax.set_xticklabels(heat.columns.tolist(), rotation=45, ha="right", fontsize=9)
    ax.set_xticklabels([])

    #ax.set_yticklabels([])
    ax.set_yticks(np.arange(len(heat.index)))
    ax.set_yticklabels([])
    #ax.set_yticklabels(heat.index.tolist(), fontsize=9)
    
    ax.tick_params(axis="x", length=6, width=2.0)
    ax.tick_params(axis="y", length=6, width=2.0)

    #ax.set_xticklabels(heat.columns.tolist(), rotation=45, ha="right", fontsize=26)
    #ax.set_yticklabels(heat.index.tolist(), fontsize=26)

    #ax.set_title(f"{variant_to_display(method_name)} | rare≤{rare_threshold:.3f} | bal acc={bal_acc_text}", fontsize=14)
    #ax.set_title(variant_to_display(method_name), fontsize=14)
    #ax.set_title(
    #f"{variant_to_display(method_name)} (Bal Acc: {bal_acc_val.mean()*100:.1f}%)",
    #fontsize=24
#)
    #ax.set_title(f"{variant_to_display(method_name)} (Bal Acc: {bal_acc_text})", fontsize=24)
    if len(bal_acc_val) > 0:
        title = f"{variant_to_display(method_name)} (Bal Acc: {bal_acc_val.mean()*100:.2f}%)"
    else:
        title = f"{variant_to_display(method_name)} (Bal Acc: NA)"

    ax.set_title(title, fontsize=24)
    #ax.set_xlabel("majority-vote label", fontsize=12)
    #ax.set_ylabel("true label", fontsize=12)

    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=10)
    #cbar.ax.tick_params(labelsize=26, width=2.0, length=6)
    ax.grid(False)

    fig.tight_layout()
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    fig.savefig(out_png.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved per-method assignment heatmap: %s", out_png)


# =========================================================
# Main driver
# =========================================================
def run_pbmc_rare_cell_experiment(
    input_h5ad: str,
    celltype_key: str,
    partition_dir: str,
    output_root: str,
    rare_thresholds: List[float],
    dpi: int = 600,
):
    dataset_name = "HumanFetalAtlas"

    safe_mkdir(output_root)
    raw_dir = os.path.join(output_root, "raw")
    summary_dir = os.path.join(output_root, "summary")
    plots_dir = os.path.join(output_root, "plots")
    for d in [raw_dir, summary_dir, plots_dir]:
        safe_mkdir(d)

    labels = load_labels(input_h5ad, celltype_key)

    rare_type_tables = []
    for rare_thr in rare_thresholds:
        rare_table = compute_rare_types(labels, rare_thr).copy()
        rare_table["dataset"] = dataset_name
        rare_table["rare_threshold"] = rare_thr
        rare_type_tables.append(rare_table)

    rare_type_all = pd.concat(rare_type_tables, ignore_index=True)
    
    # remove rare types missing in some methods
    #valid_types = None
    #for method in METHOD_ORDER:
    #    subset = rare_type_all["cell_type"].unique()
    #    if valid_types is None:
    #        valid_types = set(subset)
    #    else:
    #        valid_types = valid_types.intersection(subset)

    #rare_type_all = rare_type_all[rare_type_all["cell_type"].isin(valid_types)]
    
    #DROP_RARE_TYPES = {"Class-switched B"}
    #rare_type_all = rare_type_all[~rare_type_all["cell_type"].isin(DROP_RARE_TYPES)].copy()
    
    rare_type_csv = os.path.join(output_root, "HumanFetalAtlas_rare_cell_types.csv")
    rare_type_all.to_csv(rare_type_csv, index=False)
    logger.info("Saved rare cell type table: %s", rare_type_csv)

    all_raw = []
    all_summary = []

    for cfg in METHOD_CONFIG:
        csv_path = os.path.join(partition_dir, cfg["csv_name"])
        if not os.path.exists(csv_path):
            logger.warning("Missing CSV, skipping: %s", csv_path)
            continue

        method_name = cfg["method"]
        column_map = cfg["column_map"]

        logger.info("Processing method=%s file=%s", method_name, csv_path)

        part_df = load_partition_csv(csv_path)
        part_df = normalize_partition_columns(part_df)
        logger.info("Available columns for %s: %s", method_name, list(part_df.columns))

        for rare_thr in rare_thresholds:
            try:
                raw_df, summary_df = compute_rare_metrics_for_one_partition(
                    labels=labels,
                    partition_df=part_df,
                    dataset_name=dataset_name,
                    method_name=method_name,
                    rare_threshold=rare_thr,
                    column_map=column_map,
                )
            except Exception as e:
                logger.error("Failed for method=%s rare_threshold=%.4f: %s", method_name, rare_thr, str(e))
                continue

            raw_csv = os.path.join(
                raw_dir,
                f"{Path(cfg['csv_name']).stem}_rare{str(rare_thr).replace('.', 'p')}_raw.csv"
            )
            summary_csv = os.path.join(
                summary_dir,
                f"{Path(cfg['csv_name']).stem}_rare{str(rare_thr).replace('.', 'p')}_summary.csv"
            )

            raw_df.to_csv(raw_csv, index=False)
            summary_df.to_csv(summary_csv, index=False)

            logger.info("Saved raw: %s", raw_csv)
            logger.info("Saved summary: %s", summary_csv)

            all_raw.append(raw_df)
            all_summary.append(summary_df)

    raw_all = pd.concat(all_raw, ignore_index=True) if len(all_raw) > 0 else pd.DataFrame()
    summary_all = pd.concat(all_summary, ignore_index=True) if len(all_summary) > 0 else pd.DataFrame()

    raw_all_csv = os.path.join(output_root, "HumanFetalAtlas_rare_cell_ALL_raw.csv")
    summary_all_csv = os.path.join(output_root, "HumanFetalAtlas_rare_cell_ALL_summary.csv")
    raw_all.to_csv(raw_all_csv, index=False)
    summary_all.to_csv(summary_all_csv, index=False)

    logger.info("Saved combined raw: %s", raw_all_csv)
    logger.info("Saved combined summary: %s", summary_all_csv)

    methods_present = [m for m in METHOD_ORDER if m in set(summary_all["method"])]
    legend_png = os.path.join(plots_dir, "HumanFetalAtlas_rare_cell_legend.png")
    save_shared_legend_png(legend_png, methods_present, dpi=dpi)

    #metrics_to_plot = [
    #    "dominant_capture_mean",
    #    "best_metacell_purity_mean",
    #    "best_single_metacell_recall_mean",
    #    "capture_at_70_mean",
    #    "capture_at_90_mean",
    #]

    #metrics_to_plot = [
    #    "normalized_dominant_capture_mean",
    #    "normalized_best_single_metacell_recall_mean",
    #    "dominant_capture_mean",
    #    "best_metacell_purity_mean",
    #]

    #metrics_to_plot = [
    #    "bal_acc_rare",
    #    "normalized_dominant_capture_mean",
    #    "normalized_best_single_metacell_recall_mean",
    #    "best_metacell_purity_mean",
    #]

    metrics_to_plot = [
        "bal_acc_rare",
        "normalized_dominant_capture_mean",
        "normalized_best_single_metacell_recall_mean",
        "best_metacell_purity_mean",
        "dominant_capture_mean",
        "best_single_metacell_recall_mean",
        "capture_at_70_mean",
        "capture_at_90_mean",

        "mean_metacell_purity_mean",
        "median_metacell_purity_mean",
        "mean_metacell_recall_mean",
        "median_metacell_recall_mean",

        "normalized_mean_metacell_recall_mean",
        "normalized_median_metacell_recall_mean",


        "n_metacells_with_type_mean",
        "fragmentation_per_cell_mean",
        "rare_entropy_mean",
        "rare_entropy_normalized_mean",
        "top2_recall_mass_mean",
        "top3_recall_mass_mean",
        "best_f1_mean",

        "rare_recall_mean",
        "rare_precision_mean",
        "rare_f1_mean",
        "rare_purity_mean",
        "rare_ari",
        "rare_nmi",
    ]

    for rare_thr in rare_thresholds:
        thr_dir = os.path.join(plots_dir, f"rare_{str(rare_thr).replace('.', 'p')}")
        safe_mkdir(thr_dir)

        for metric in metrics_to_plot:
            out_png_m = os.path.join(
                thr_dir,
                f"HumanFetalAtlas_{metric}_vs_nmetacells_rare{str(rare_thr).replace('.', 'p')}.png"
            )
            plot_metric(
                summary_df=summary_all,
                metric_col=metric,
                rare_threshold=rare_thr,
                out_png=out_png_m,
                dpi=dpi,
            )

    celltype_plot_dir = os.path.join(plots_dir, "rare_celltype_boxplots")
    safe_mkdir(celltype_plot_dir)


    combined_boxplot_dir = os.path.join(plots_dir, "combined_method_boxplots")
    safe_mkdir(combined_boxplot_dir)

    combined_legend_png = os.path.join(combined_boxplot_dir, "HumanFetalAtlas_grouped_boxplot_legend.png")
    save_shared_legend_png(combined_legend_png, methods_present, dpi=dpi)

    #for rare_thr in rare_thresholds:
    #    thr_dir = os.path.join(celltype_plot_dir, f"rare_{str(rare_thr).replace('.', 'p')}")
    #    safe_mkdir(thr_dir)

    #    for method in methods_present:
    #        #for metric in metrics_to_plot:
    #        for metric in ["best_metacell_purity","mean_metacell_purity","median_metacell_purity", "dominant_capture", "best_single_metacell_recall","mean_metacell_recall","median_metacell_recall", "normalized_dominant_capture", "normalized_best_single_metacell_recall", "normalized_mean_metacell_recall","normalized_median_metacell_recall",                "n_metacells_with_type",
    #            "fragmentation_per_cell",
    #            "rare_entropy",
    #            "rare_entropy_normalized",
    #            "top2_recall_mass",
    #            "top3_recall_mass",
    #            "best_f1",]:
    #            out_png = os.path.join(
    #                thr_dir,
    #                f"PBMC_{method}_{metric}_rare{str(rare_thr).replace('.', 'p')}.png"
    #            )
    #            plot_per_method_rare_celltype_boxplots(
    #                raw_df=raw_all,
    #                rare_type_df=rare_type_all,
    #                metric_col=metric,
    #                rare_threshold=rare_thr,
    #                method_name=method,
    #                out_png=out_png,
    #                dpi=dpi,
    #            )

    for rare_thr in rare_thresholds:
        thr_dir = os.path.join(combined_boxplot_dir, f"rare_{str(rare_thr).replace('.', 'p')}")
        safe_mkdir(thr_dir)

        for metric in [
            "best_metacell_purity",
            "mean_metacell_purity",
            "median_metacell_purity",
            "best_single_metacell_recall",
            "mean_metacell_recall",
            "median_metacell_recall",
            "normalized_best_single_metacell_recall",
            "normalized_mean_metacell_recall",
            "normalized_median_metacell_recall",
            "dominant_capture",
            "normalized_dominant_capture",
            "capture_at_70",
            "capture_at_90",
            "n_metacells_with_type",
            "fragmentation_per_cell",
            "rare_entropy",
            "rare_entropy_normalized",
            "top2_recall_mass",
            "top3_recall_mass",
            "best_f1",

            "rare_recall",
            "rare_precision",
            "rare_f1",
            "rare_purity",

            "rare_recall_mean",
            "rare_precision_mean",
            "rare_f1_mean",
            "rare_purity_mean",
        ]:
            out_png = os.path.join(
                thr_dir,
                f"HumanFetalAtlas_grouped_celltype_methods_{metric}_rare{str(rare_thr).replace('.', 'p')}.png"
            )
            plot_grouped_celltype_method_boxplot(
                raw_df=raw_all,
                rare_type_df=rare_type_all,
                rare_threshold=rare_thr,
                metric_col=metric,
                out_png=out_png,
                dpi=dpi,
            )


    #for rare_thr in rare_thresholds:
    #    thr_dir = os.path.join(combined_boxplot_dir, f"rare_{str(rare_thr).replace('.', 'p')}")
    #    safe_mkdir(thr_dir)

    #    for metric in [
    #        "best_metacell_purity",
    #        "best_single_metacell_recall",
    #        "normalized_best_single_metacell_recall",
    #        "dominant_capture",
    #        "normalized_dominant_capture",
    #        "capture_at_70",
    #        "capture_at_90",
    #    ]:
    #        out_png = os.path.join(
    #            thr_dir,
    #            f"PBMC_combined_methods_{metric}_rare{str(rare_thr).replace('.', 'p')}.png"
    #        )
    #        plot_combined_method_boxplot(
    #            raw_df=raw_all,
    #            rare_threshold=rare_thr,
    #            metric_col=metric,
    #            out_png=out_png,
    #            dpi=dpi,
    #        )


    #for rare_thr in rare_thresholds:
    #    thr_dir = os.path.join(combined_boxplot_dir, f"rare_{str(rare_thr).replace('.', 'p')}")
    #    safe_mkdir(thr_dir)

    #    for metric in [
    #        "best_metacell_purity",
    #        "best_single_metacell_recall",
    #        "normalized_best_single_metacell_recall",
    #        "dominant_capture",
    #        "normalized_dominant_capture",
    #        "capture_at_70",
    #        "capture_at_90",
    #    ]:
    #        out_png = os.path.join(
    #            thr_dir,
    #            f"PBMC_combined_methods_{metric}_rare{str(rare_thr).replace('.', 'p')}.png"
    #        )
    #        plot_combined_method_boxplot(
    #            raw_df=raw_all,
    #            rare_threshold=rare_thr,
    #            metric_col=metric,
    #            out_png=out_png,
    #            dpi=dpi,
    #        )



    heatmap_dir = os.path.join(plots_dir, "assignment_heatmaps")
    safe_mkdir(heatmap_dir)

    for rare_thr in rare_thresholds:
        thr_dir = os.path.join(heatmap_dir, f"rare_{str(rare_thr).replace('.', 'p')}")
        safe_mkdir(thr_dir)

        for method in methods_present:
            out_png = os.path.join(
                thr_dir,
                f"HumanFetalAtlas_{method}_assignment_heatmap_rare{str(rare_thr).replace('.', 'p')}.png"
            )
            plot_per_method_assignment_heatmap(
                raw_df=raw_all,
                rare_type_df=rare_type_all,
                rare_threshold=rare_thr,
                method_name=method,
                out_png=out_png,
                dpi=dpi,
            )


def main():
    #INPUT_H5AD = "/storage/home/dvl5760/scratch/blish_covid.seu.h5ad"
    #CELLTYPE_KEY = "cell.type"
    #PARTITION_DIR = "/storage/home/dvl5760/scratch/partitions_covid_healthy"
    #OUTPUT_ROOT = "/storage/home/dvl5760/work/camp_gr/rare_cell_pbmc_results"

    #RARE_THRESHOLDS = [0.01, 0.03, 0.05]
    #RARE_THRESHOLDS = [0.01]
    #DPI = 300

    INPUT_H5AD = "/storage/home/dvl5760/scratch/GSE156793_scjoint_subsample_prep.h5ad"
    CELLTYPE_KEY = "Main_cluster_name"
    PARTITION_DIR = "/storage/home/dvl5760/scratch/partitions"
    OUTPUT_ROOT = "/storage/home/dvl5760/work/camp_gr/rare_cell_human_fetal_atlas_results"

    RARE_THRESHOLDS = [0.001]
    DPI = 300

    logger.info("Starting PBMC rare-cell experiment")
    logger.info(json.dumps({
        "INPUT_H5AD": INPUT_H5AD,
        "CELLTYPE_KEY": CELLTYPE_KEY,
        "PARTITION_DIR": PARTITION_DIR,
        "OUTPUT_ROOT": OUTPUT_ROOT,
        "RARE_THRESHOLDS": RARE_THRESHOLDS,
        "DPI": DPI,
    }, indent=2))

    run_pbmc_rare_cell_experiment(
        input_h5ad=INPUT_H5AD,
        celltype_key=CELLTYPE_KEY,
        partition_dir=PARTITION_DIR,
        output_root=OUTPUT_ROOT,
        rare_thresholds=RARE_THRESHOLDS,
        dpi=DPI,
    )

    logger.info("PBMC rare-cell experiment finished.")


if __name__ == "__main__":
    main()
