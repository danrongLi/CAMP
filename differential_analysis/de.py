#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dataset-aware adaptations of MetaQ Fig. 4 (batch integration) and Fig. 5
(differential-expression preservation) for CAMP and released baselines.

The pipeline follows CAMP's PBMC preprocessing (cell/gene filtering,
library-size normalization, log1p, 2,000 HVGs, scaling, and PCA) and saves
plot-ready CSV checkpoints. Native released partitions are retained as an
audit. The primary controlled comparison minimally merges intact native
metacells to exactly 750, 1,000, and 1,250 profiles for Fig. 4. For Fig. 5,
every method is rebalanced only within cell-type x donor x condition strata to
the same exact profile budget at 8x, 10x, and 12x compression. Thus neither a
method's global resolution grid nor its number of post-stratification fragments
can provide an unequal profile-count advantage. Re-run with ``--stage plot``
to redraw figures without repeating computation.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import logging
import os
import random
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

# Repository copy used by CAMP4 without installing the full SEACells package.
LOCAL_SEACELLS_SOURCE = Path(__file__).resolve().parent / "SEACells"
LOCAL_SEACELLS_BUILD_GRAPH = LOCAL_SEACELLS_SOURCE / "SEACells" / "build_graph.py"

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("MKL_DYNAMIC", "FALSE")

import anndata as ad
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch, Rectangle
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from scipy import sparse
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix
from scipy.stats import kendalltau, pearsonr, rankdata, spearmanr, ttest_ind
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    balanced_accuracy_score,
    f1_score,
    homogeneity_score,
    silhouette_score,
)
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.preprocessing import normalize


# =========================================================
# Logging / style
# =========================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
logger = logging.getLogger(__name__)

DISPLAY_NAME_MAP = {
    "camp1": "CAMP1",
    "camp2": "CAMP2",
    "camp3": "CAMP3",
    "camp4": "CAMP4",
    "original": "Original cells",
    "seacells": "SEACells",
    "metacell1": "MetaCell1",
    "metacell2": "MetaCell2",
    "supercell": "SuperCell",
    "metaq": "MetaQ",
}

CUSTOM_PALETTE = {
    "Original cells": "#111111",
    "CAMP1": "#1f77b4",
    "CAMP2": "#ff7f0e",
    "CAMP3": "#2ca02c",
    "CAMP4": "#bcbd22",
    "SEACells": "#d62728",
    "SuperCell": "#9467bd",
    "MetaCell": "#8c564b",
    "MetaCell1": "#8c564b",
    "MetaCell2": "#e377c2",
    "MetaQ": "#7f7f7f",
}

# One method identity is used in every main-text, supplementary, and MetaQ-style
# comparison. MetaCell1 is retained as the internal/checkpoint name because it
# is the first-generation MetaCell implementation; publication panels display
# it as "MetaCell", matching the manuscript terminology.
PUBLICATION_METHOD_LABELS = {
    "MetaCell1": "MetaCell",
    "Original cells": "Full cells",
    "Harmony": "Full cells",
}
PUBLICATION_METHOD_ORDER = [
    "CAMP1",
    "CAMP2",
    "CAMP3",
    "CAMP4",
    "SEACells",
    "SuperCell",
    "MetaCell",
    "MetaCell2",
    "MetaQ",
]
DEFAULT_FOCUSED_METHOD_ORDER = [
    "CAMP1",
    "SEACells",
    "SuperCell",
    "MetaCell",
    "MetaCell2",
    "MetaQ",
]
PUBLICATION_PALETTE = {
    "CAMP1": "#1f77b4",
    "CAMP2": "#ff7f0e",
    "CAMP3": "#2ca02c",
    "CAMP4": "#bcbd22",
    "SEACells": "#d62728",
    "SuperCell": "#9467bd",
    "MetaCell": "#8c564b",
    "MetaCell2": "#e377c2",
    "MetaQ": "#7f7f7f",
    "Full cells": "#111111",
}

# Clinical-status colors deliberately do not reuse the blue CAMP1 or red
# SEACells identities.  Keeping biological metadata and method identities on
# separate visual palettes prevents the status UMAP from implying a method.
DE_STATUS_PALETTE = {
    "Healthy": "#007C91",
    "COVID": "#E69F00",
}
DE_EXAMPLE_HIGHLIGHT_COLOR = "#00A651"
DE_EXAMPLE_EXTERNAL_COMPARATORS = {
    "SEACells",
    "SuperCell",
    "MetaCell",
    "MetaCell2",
    "MetaQ",
}

# These panels are placed as subfigures in the manuscript, so their source
# fonts must be substantially larger than matplotlib defaults to remain
# readable after LaTeX scales the PDFs down.
DE_AXIS_LABEL_FONTSIZE = 22
DE_TICK_FONTSIZE = 18
DE_TITLE_FONTSIZE = 23
DE_HEATMAP_VALUE_FONTSIZE = 17

# MetaQ-style panels use the same manuscript palette instead of a second visual
# identity. This makes a method's color invariant across every experiment.
METAQ_PAPER_FOCAL_COLOR = PUBLICATION_PALETTE["CAMP1"]
METAQ_PAPER_COMPARATOR_PALETTE = {
    "Original cells": "#111111",
    "Harmony": "#111111",
    "MetaQ": "#7f7f7f",
    "SEACells": "#d62728",
    "MetaCell1": "#8c564b",
    "MetaCell2": "#e377c2",
    "SuperCell": "#9467bd",
}
METAQ_PAPER_BASELINES = ["MetaQ", "SEACells", "MetaCell1", "MetaCell2", "SuperCell"]
ALL_METHOD_ORDER = [
    "CAMP1",
    "CAMP2",
    "CAMP3",
    "CAMP4",
    "MetaQ",
    "SEACells",
    "MetaCell1",
    "MetaCell2",
    "SuperCell",
]
BLINEAGE_METHOD_PALETTE = {
    "Harmony": "#111111",
    "CAMP1": "#1f77b4",
    "CAMP2": "#ff7f0e",
    "CAMP3": "#2ca02c",
    "CAMP4": "#bcbd22",
    "MetaQ": "#7f7f7f",
    "SEACells": "#d62728",
    "MetaCell1": "#8c564b",
    "MetaCell2": "#e377c2",
    "SuperCell": "#9467bd",
}

NATIVE_DE_PROTOCOL_VERSION = "native_partition_stratum_intersection_grid_v2"
MATCHED_DE_PROTOCOL_VERSION = "matched_within_stratum_reduction_grid_v3"
FIG4_PROTOCOL_VERSION = "native_released_resolution_grid_v2"
MATCHED_FIG4_PROTOCOL_VERSION = "matched_count_native_fragment_coarsening_grid_v3"
BLINEAGE_PROTOCOL_VERSION = "donor_blocked_blineage_matched_count_grid_v2"

# Native PBMC memberships produced by the released CAMP work tree. All numeric
# resolution columns are considered and a one-to-one closest native grid is
# selected across the requested resolutions.
RELEASED_PBMC_BASELINES = {
    "SEACells": {
        "relative_path": "seacell_default_output/covid_healthy/seacell_default_partition.csv",
        "parameter_name": "gamma",
    },
    "MetaCell1": {
        "relative_path": "customized_metacell/from_local_to_server_methods/covid_healthy/metacell1_membership_amp.csv",
        "parameter_name": "initial_knn_amp",
    },
    "MetaCell2": {
        "relative_path": "customized_metacell/data/covid_healthy/metacell2_membership_small_gamma.csv",
        "parameter_name": "target_metacell_size",
    },
    "SuperCell": {
        "relative_path": "customized_metacell/from_local_to_server_methods/covid_healthy/supercell_membership.csv",
        "parameter_name": "requested_supercells",
    },
    "MetaQ": {
        "relative_path": "customized_metacell/MetaQ/save/combined_metacell_labels.csv",
        "parameter_name": "metacell_num",
    },
}

# Full-atlas memberships produced by the released CAMP HFA work tree.  CAMP1-3
# are the outputs of the repository's sparse, on-the-fly HFA implementations;
# CAMP4 and the competing methods are their released native outputs.  Every
# numeric resolution column is eligible for the same exact-count merge-only
# comparison used below.  MetaCell1/2 are intentionally absent: the uploaded
# work tree contains only 17,824- and 5,000-cell partial HFA memberships,
# respectively, rather than partitions of the ~494,600-cell evaluation atlas.
RELEASED_HFA_BASELINES = {
    "CAMP1": {
        "relative_path": "customized_metacell/human_fetal_atlas/output/edit_4_partitions.csv",
        "parameter_name": "gamma",
    },
    "CAMP2": {
        "relative_path": "customized_metacell/human_fetal_atlas/output/edit_4_add_simi_partitions.csv",
        "parameter_name": "gamma",
    },
    "CAMP3": {
        "relative_path": "customized_metacell/human_fetal_atlas/output/edit_4_add_ad_gau_partitions.csv",
        "parameter_name": "gamma",
    },
    "CAMP4": {
        "relative_path": "customized_metacell/human_fetal_atlas/output/edit_5_partitions_full_metacell.csv",
        "parameter_name": "gamma",
    },
    "SEACells": {
        "relative_path": "seacell_default_output/human_fetal_atlas/seacell_default_partition.csv",
        "parameter_name": "gamma",
    },
    "SuperCell": {
        "relative_path": "customized_metacell/from_local_to_server_methods/human_supercell/supercell_membership_approx_new.csv",
        "parameter_name": "requested_supercells",
    },
    "MetaQ": {
        "relative_path": "customized_metacell/MetaQ/save/human_fetal_atlas_combined_metacell_labels.csv",
        "parameter_name": "metacell_num",
    },
}

HFA_UNAVAILABLE_BASELINES = {
    "MetaCell1": (
        "No full-atlas released membership: the available CSV contains 17,824 "
        "rows and cannot be fairly compared on the shared ~494,600-cell universe."
    ),
    "MetaCell2": (
        "No full-atlas released membership: the available CSV contains 5,000 "
        "rows and cannot be fairly compared on the shared ~494,600-cell universe."
    ),
}

# This variable is selected from the dataset name in main().  Keeping one
# mapping lets the mature native-grid/matched-count code serve both datasets.
RELEASED_BASELINES = RELEASED_PBMC_BASELINES

CELLTYPE_CANDIDATES = [
    "cell.type",
    "cell.type.coarse",
    "celltype",
    "cell_type",
    "CellType",
    "Main_cluster_name",
]
BATCH_CANDIDATES = [
    "orig.ident",
    "sample",
    "Sample",
    "sample_id",
    "donor",
    "Donor",
    "patient",
    "Patient",
    "batch",
    "Batch",
]
DONOR_CANDIDATES = [
    "donor",
    "Donor",
    "patient",
    "Patient",
    "orig.ident",
    "sample",
    "Sample",
    "sample_id",
]
CONDITION_CANDIDATES = [
    "Status",
    "status",
    "condition",
    "Condition",
    "perturbation",
    "perturbation_name",
    "disease",
    "disease_status",
]
CONTROL_NAMES = ["Healthy", "healthy", "Control", "control", "DMSO", "dmso", "NT", "negative control"]


# =========================================================
# Small utilities
# =========================================================
def safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def slugify(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("_")
    return value or "unnamed"


def display_name(value: str) -> str:
    return DISPLAY_NAME_MAP.get(str(value).lower(), str(value))


def publication_method_label(value: object) -> str:
    """Return the manuscript label without changing checkpoint identities."""
    value = str(value)
    return PUBLICATION_METHOD_LABELS.get(value, value)


def publication_method_order(values: Iterable[object], include_full_cells: bool = True) -> List[str]:
    observed = {publication_method_label(value) for value in values}
    preferred = list(PUBLICATION_METHOD_ORDER)
    if include_full_cells:
        preferred.append("Full cells")
    ordered = [method for method in preferred if method in observed]
    ordered.extend(sorted(observed.difference(ordered)))
    return ordered


def publication_method_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Copy a result table and standardize only its displayed method labels."""
    result = frame.copy()
    if "method" in result.columns:
        result["method"] = result["method"].map(publication_method_label)
    return result


def save_csv(df: pd.DataFrame, path: Path, index: bool = False) -> None:
    safe_mkdir(path.parent)
    df.to_csv(path, index=index)
    logger.info("Saved CSV: %s | rows=%d", path, len(df))


def save_json(obj: Mapping, path: Path) -> None:
    safe_mkdir(path.parent)
    with path.open("w") as handle:
        json.dump(obj, handle, indent=2, default=str)
    logger.info("Saved metadata: %s", path)


def read_csvs(paths: Iterable[Path]) -> pd.DataFrame:
    paths = sorted(paths)
    if not paths:
        return pd.DataFrame()
    return pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def load_seacells_build_graph():
    """Load only SEACells.build_graph, avoiding unrelated optional imports."""
    if LOCAL_SEACELLS_BUILD_GRAPH.is_file():
        module_name = "camp_local_seacells_build_graph"
        if module_name in sys.modules:
            return sys.modules[module_name]
        spec = importlib.util.spec_from_file_location(
            module_name,
            LOCAL_SEACELLS_BUILD_GRAPH,
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load {LOCAL_SEACELLS_BUILD_GRAPH}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module

    try:
        from SEACells import build_graph
    except ImportError as exc:
        raise ImportError(
            "CAMP4 requires SEACells. Upload the local SEACells folder next to "
            "camp_pbmc_fig4_fig5.py or install SEACells in the active environment."
        ) from exc
    return build_graph


def clip_negative_sparse_inplace(X) -> int:
    if sparse.issparse(X):
        mask = X.data < 0
        n_negative = int(mask.sum())
        if n_negative:
            X.data[mask] = 0.0
            X.eliminate_zeros()
        return n_negative
    n_negative = int(np.sum(X < 0))
    if n_negative:
        X[:] = np.maximum(X, 0.0)
    return n_negative


def get_total_counts(X) -> np.ndarray:
    return np.asarray(X.sum(axis=1)).ravel()


def resolve_obs_key(
    obs: pd.DataFrame,
    requested: str,
    candidates: Sequence[str],
    purpose: str,
    required: bool = True,
) -> Optional[str]:
    if requested != "auto":
        if requested not in obs.columns:
            raise KeyError(
                f"Requested {purpose} key '{requested}' is absent. "
                f"Available keys: {list(obs.columns)}"
            )
        return requested

    for candidate in candidates:
        if candidate in obs.columns:
            logger.info("Auto-detected %s key: %s", purpose, candidate)
            return candidate

    if required:
        raise KeyError(
            f"Could not auto-detect a {purpose} key. Pass --{purpose.replace('_', '-')}-key. "
            f"Available keys: {list(obs.columns)}"
        )
    return None


def clean_obs_values(values: pd.Series) -> pd.Series:
    values = values.astype("string").fillna("Missing").astype(str)
    values[values.isin(["nan", "None", "<NA>"])] = "Missing"
    return values


def choose_reference_condition(values: pd.Series, requested: str) -> str:
    unique_values = clean_obs_values(values).unique().tolist()
    if requested != "auto":
        if requested not in unique_values:
            raise ValueError(
                f"Reference condition '{requested}' is absent. Found: {unique_values}"
            )
        return requested
    for name in CONTROL_NAMES:
        if name in unique_values:
            logger.info("Auto-detected reference condition: %s", name)
            return name
    reference = clean_obs_values(values).value_counts().index[0]
    logger.warning(
        "No standard control label was found; using the most frequent condition '%s' as reference.",
        reference,
    )
    return str(reference)


# =========================================================
# CAMP PBMC preprocessing and cache
# =========================================================
def read_dataset(input_file: str, loom_var_name_key: str) -> ad.AnnData:
    path = Path(input_file)
    logger.info("Reading dataset: %s", path)
    if path.suffix.lower() == ".h5ad":
        adata = sc.read_h5ad(path)
    elif path.suffix.lower() == ".loom":
        adata = sc.read_loom(path, sparse=True, dtype="float32")
    else:
        raise ValueError("Input must be .h5ad or .loom")

    if loom_var_name_key in adata.var.columns:
        adata.var_names = adata.var[loom_var_name_key].astype(str)
    else:
        adata.var_names = adata.var_names.astype(str)
    adata.var_names_make_unique()
    adata.obs_names = adata.obs_names.astype(str)
    adata.var.index.name = None
    return adata


def build_or_load_preprocessing_cache(
    args: argparse.Namespace,
    cache_dir: Path,
) -> Tuple[ad.AnnData, ad.AnnData]:
    dataset_slug = slugify(args.dataset_name).lower()
    base_path = cache_dir / f"{dataset_slug}_log1p_full_genes.h5ad"
    model_path = cache_dir / f"{dataset_slug}_hvg{args.n_hvg}_pca{args.n_pcs}.h5ad"
    metadata_path = cache_dir / "preprocessing_summary.csv"

    if base_path.exists() and model_path.exists() and not args.force:
        logger.info("Using cached preprocessing: %s and %s", base_path, model_path)
        base = sc.read_h5ad(base_path)
        model = sc.read_h5ad(model_path)
        if not base.obs_names.equals(model.obs_names):
            raise RuntimeError("Cached full-gene and HVG objects have different cell orders")
        return base, model

    start = time.time()
    adata = read_dataset(args.input_h5ad, args.loom_var_name_key)
    raw_shape = adata.shape
    logger.info("Raw shape: %s", raw_shape)

    sc.pp.filter_cells(adata, min_genes=args.min_genes)
    sc.pp.filter_genes(adata, min_cells=args.min_cells)
    filtered_shape = adata.shape
    logger.info("After filtering: %s", filtered_shape)

    n_negative = clip_negative_sparse_inplace(adata.X)
    if n_negative:
        logger.warning("Clipped %d negative values before normalization", n_negative)

    totals = get_total_counts(adata.X)
    if np.any(totals <= 0):
        raise RuntimeError("At least one cell has zero total counts after filtering")
    sc.pp.normalize_total(adata, target_sum=args.normalize_target_sum)
    sc.pp.log1p(adata)
    if sparse.issparse(adata.X):
        adata.X = adata.X.tocsr().astype(np.float32)
    else:
        adata.X = np.asarray(adata.X, dtype=np.float32)

    gene_means = np.asarray(adata.X.mean(axis=0)).ravel()
    finite = np.isfinite(gene_means)
    if not finite.all():
        logger.warning("Removing %d genes with non-finite means", int((~finite).sum()))
        adata = adata[:, finite].copy()
    if adata.n_vars == 0:
        raise RuntimeError("No genes remain after preprocessing")

    base = adata.copy()
    base.write_h5ad(base_path, compression="gzip")
    logger.info("Saved full-gene log-normalized cache: %s", base_path)

    model = adata.copy()
    n_hvg_use = min(args.n_hvg, model.n_vars)
    sc.pp.highly_variable_genes(model, n_top_genes=n_hvg_use, flavor="seurat")
    model = model[:, model.var["highly_variable"]].copy()
    n_pcs_use = min(args.n_pcs, model.n_obs - 1, model.n_vars - 1)
    if n_pcs_use < 2:
        raise RuntimeError("Too few cells or genes to compute PCA")
    if dataset_slug == "human_fetal_atlas":
        # Match CAMP/on_the_fly_for_human_fetal_atlas_data: retain the sparse
        # log-HVG matrix, do not densify with scaling, and use uncentered PCA.
        sc.tl.pca(
            model,
            n_comps=n_pcs_use,
            svd_solver="arpack",
            zero_center=False,
            random_state=args.random_seed,
        )
        pca_preprocessing = "sparse_log_hvg_uncentered_pca"
    else:
        # Preserve the original PBMC pipeline exactly.
        sc.pp.scale(model, max_value=10)
        sc.tl.pca(
            model,
            n_comps=n_pcs_use,
            svd_solver="arpack",
            random_state=args.random_seed,
        )
        pca_preprocessing = "scaled_log_hvg_centered_pca"
    model.write_h5ad(model_path, compression="gzip")
    logger.info("Saved HVG/PCA cache: %s", model_path)

    summary = pd.DataFrame(
        [
            {
                "source_file": str(Path(args.input_h5ad).resolve()),
                "dataset_name": args.dataset_name,
                "raw_cells": raw_shape[0],
                "raw_genes": raw_shape[1],
                "filtered_cells": filtered_shape[0],
                "filtered_genes": filtered_shape[1],
                "full_genes_after_cleaning": base.n_vars,
                "hvg_genes": model.n_vars,
                "pca_dimensions": model.obsm["X_pca"].shape[1],
                "min_genes": args.min_genes,
                "min_cells": args.min_cells,
                "normalize_target_sum": args.normalize_target_sum,
                "negative_values_clipped": n_negative,
                "pca_preprocessing": pca_preprocessing,
                "seconds": time.time() - start,
            }
        ]
    )
    save_csv(summary, metadata_path)
    return base, model


# =========================================================
# Metacell aggregation
# =========================================================
def majority_by_code(values: pd.Series, codes: np.ndarray, n_groups: int) -> List[str]:
    clean = clean_obs_values(values).to_numpy()
    output: List[str] = []
    for group in range(n_groups):
        group_values = clean[codes == group]
        if len(group_values) == 0:
            output.append("Missing")
        else:
            labels, counts = np.unique(group_values, return_counts=True)
            output.append(str(labels[np.argmax(counts)]))
    return output


def aggregate_metacells(
    adata: ad.AnnData,
    assignment: pd.Series,
    obs_keys: Mapping[str, str],
    method: str,
) -> Tuple[ad.AnnData, np.ndarray]:
    assignment = assignment.reindex(adata.obs_names)
    if assignment.isna().any():
        raise ValueError(f"{method}: assignment cannot be aligned to AnnData cells")
    raw_ids = assignment.astype(str).to_numpy()
    unique_ids, codes = np.unique(raw_ids, return_inverse=True)
    n_metacells = len(unique_ids)

    rows = codes
    columns = np.arange(adata.n_obs)
    sizes = np.bincount(codes, minlength=n_metacells).astype(np.float64)
    membership = csr_matrix(
        (1.0 / sizes[codes], (rows, columns)),
        shape=(n_metacells, adata.n_obs),
        dtype=np.float32,
    )
    X_meta = membership @ adata.X
    if sparse.issparse(X_meta):
        X_meta = X_meta.tocsr().astype(np.float32)
    else:
        X_meta = np.asarray(X_meta, dtype=np.float32)

    meta = ad.AnnData(X=X_meta, var=adata.var.copy())
    meta.obs_names = pd.Index([f"{slugify(method)}__mc_{i}" for i in range(n_metacells)])
    meta.obs["metacell_id"] = unique_ids
    meta.obs["n_cells"] = sizes.astype(int)
    meta.obs["method"] = method
    for canonical_name, source_key in obs_keys.items():
        meta.obs[canonical_name] = majority_by_code(
            adata.obs[source_key], codes, n_metacells
        )
    return meta, codes


# =========================================================
# Fig. 4: Harmony, mapping, clustering, and LISI
# =========================================================
def pca_for_metacells(meta: ad.AnnData, n_pcs: int, seed: int) -> np.ndarray:
    n_comps = min(n_pcs, meta.n_obs - 1, meta.n_vars - 1)
    if n_comps < 2:
        raise RuntimeError("Too few metacells to compute PCA")
    sc.tl.pca(meta, n_comps=n_comps, svd_solver="arpack", random_state=seed)
    return np.asarray(meta.obsm["X_pca"], dtype=np.float32)


def harmony_integrate(
    embedding: np.ndarray,
    obs: pd.DataFrame,
    batch_key: str,
    max_iter: int,
) -> np.ndarray:
    if clean_obs_values(obs[batch_key]).nunique() < 2:
        logger.warning("Only one batch is present; returning uncorrected PCA")
        return np.asarray(embedding, dtype=np.float32)
    try:
        import harmonypy
    except ImportError as exc:
        raise ImportError(
            "Fig. 4 requires harmonypy==0.0.6 to match the MetaQ Harmony "
            "protocol. Activate the job environment and run: "
            "python -m pip install harmonypy==0.0.6"
        ) from exc

    metadata = obs.copy()
    metadata[batch_key] = clean_obs_values(metadata[batch_key])
    nclust = min(15, max(2, embedding.shape[0] // 20))
    logger.info(
        "Running Harmony | cells=%d dims=%d batches=%d nclust=%d",
        embedding.shape[0],
        embedding.shape[1],
        metadata[batch_key].nunique(),
        nclust,
    )
    result = harmonypy.run_harmony(
        embedding,
        metadata,
        batch_key,
        max_iter_harmony=max_iter,
        nclust=nclust,
        theta=10.0,
        verbose=False,
    )
    return np.ascontiguousarray(result.Z_corr.T, dtype=np.float32)


def mapping_network(
    X_train,
    target: np.ndarray,
    X_all,
    epochs: int,
    batch_size: int,
    device_name: str,
    seed: int,
) -> Tuple[np.ndarray, pd.DataFrame]:
    import torch
    from torch import nn

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but unavailable; using CPU")
        device_name = "cpu"
    device = torch.device(device_name)

    def dense_float32(X):
        if sparse.issparse(X):
            return X.toarray().astype(np.float32, copy=False)
        return np.asarray(X, dtype=np.float32)

    train_x = torch.from_numpy(np.ascontiguousarray(dense_float32(X_train)))
    train_y = torch.from_numpy(np.ascontiguousarray(target, dtype=np.float32))
    model = nn.Sequential(
        nn.Linear(train_x.shape[1], 256),
        nn.ReLU(),
        nn.Linear(256, train_y.shape[1]),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    generator = torch.Generator().manual_seed(seed)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_x, train_y),
        batch_size=min(batch_size, len(train_x)),
        shuffle=True,
        generator=generator,
    )

    history = []
    logger.info(
        "Training raw-to-Harmony mapping | input=%d hidden=256 output=%d epochs=%d device=%s",
        train_x.shape[1],
        train_y.shape[1],
        epochs,
        device,
    )
    model.train()
    for epoch in range(1, epochs + 1):
        running = 0.0
        seen = 0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch_x)
            loss = loss_fn(prediction, batch_y)
            loss.backward()
            optimizer.step()
            running += float(loss.detach().cpu()) * len(batch_x)
            seen += len(batch_x)
        epoch_loss = running / max(1, seen)
        history.append({"epoch": epoch, "mse": epoch_loss})
        if epoch == 1 or epoch == epochs or epoch % max(1, epochs // 10) == 0:
            logger.info("Mapping epoch %d/%d | MSE=%.6f", epoch, epochs, epoch_loss)

    model.eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, X_all.shape[0], batch_size):
            block = dense_float32(X_all[start : start + batch_size])
            block_tensor = torch.from_numpy(block).to(device)
            predictions.append(model(block_tensor).cpu().numpy())
    return np.vstack(predictions).astype(np.float32), pd.DataFrame(history)


def embedding_umap(
    embedding: np.ndarray,
    obs: pd.DataFrame,
    seed: int,
    n_neighbors: int,
) -> pd.DataFrame:
    if embedding.shape[0] < 3:
        coords = np.column_stack([np.arange(embedding.shape[0]), np.zeros(embedding.shape[0])])
    else:
        temp = ad.AnnData(X=np.asarray(embedding, dtype=np.float32), obs=obs.copy())
        sc.pp.neighbors(
            temp,
            use_rep="X",
            n_neighbors=min(n_neighbors, temp.n_obs - 1),
            metric="cosine",
            random_state=seed,
        )
        sc.tl.umap(temp, random_state=seed)
        coords = temp.obsm["X_umap"]
    output = obs.copy().reset_index(names="item_id")
    output["UMAP1"] = coords[:, 0]
    output["UMAP2"] = coords[:, 1]
    return output


def cluster_embedding(
    embedding: np.ndarray,
    resolutions: Sequence[float],
    seed: int,
    n_neighbors: int,
) -> Tuple[pd.DataFrame, str]:
    temp = ad.AnnData(X=np.asarray(embedding, dtype=np.float32))
    sc.pp.neighbors(
        temp,
        use_rep="X",
        n_neighbors=min(n_neighbors, temp.n_obs - 1),
        metric="cosine",
        random_state=seed,
    )
    algorithm = "louvain"
    output = pd.DataFrame(index=np.arange(temp.n_obs))
    for resolution in resolutions:
        key = f"cluster_{resolution:g}"
        try:
            sc.tl.louvain(temp, resolution=float(resolution), key_added=key, random_state=seed)
        except Exception as exc:
            algorithm = "leiden_fallback"
            logger.warning("Louvain unavailable (%s); using Leiden fallback", exc)
            sc.tl.leiden(temp, resolution=float(resolution), key_added=key, random_state=seed)
        output[str(resolution)] = temp.obs[key].astype(str).to_numpy()
    return output, algorithm


def score_clusters(
    cluster_frame: pd.DataFrame,
    truth: Sequence[str],
    method: str,
    representation: str,
    algorithm: str,
    expansion_codes: Optional[np.ndarray] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    truth = np.asarray(truth).astype(str)
    metrics = []
    assignments = []
    for resolution in cluster_frame.columns:
        labels = cluster_frame[resolution].to_numpy()
        if expansion_codes is not None:
            labels = labels[expansion_codes]
        if len(labels) != len(truth):
            raise ValueError("Cluster labels and ground truth have different lengths")
        metrics.extend(
            [
                {
                    "method": method,
                    "representation": representation,
                    "algorithm": algorithm,
                    "resolution": float(resolution),
                    "metric": "AMI",
                    "score": adjusted_mutual_info_score(truth, labels),
                },
                {
                    "method": method,
                    "representation": representation,
                    "algorithm": algorithm,
                    "resolution": float(resolution),
                    "metric": "ARI",
                    "score": adjusted_rand_score(truth, labels),
                },
                {
                    "method": method,
                    "representation": representation,
                    "algorithm": algorithm,
                    "resolution": float(resolution),
                    "metric": "Homogeneity",
                    "score": homogeneity_score(truth, labels),
                },
            ]
        )
        assignments.append(
            pd.DataFrame(
                {
                    "cell_position": np.arange(len(labels)),
                    "method": method,
                    "representation": representation,
                    "resolution": float(resolution),
                    "cluster": labels,
                }
            )
        )
    return pd.DataFrame(metrics), pd.concat(assignments, ignore_index=True)


def local_lisi_fallback(
    embedding: np.ndarray,
    labels: Sequence[str],
    perplexity: int,
) -> np.ndarray:
    n_neighbors = min(max(2, 3 * perplexity), embedding.shape[0] - 1)
    neighbors = NearestNeighbors(n_neighbors=n_neighbors + 1, metric="euclidean").fit(embedding)
    indices = neighbors.kneighbors(return_distance=False)[:, 1:]
    labels = np.asarray(labels).astype(str)
    scores = np.empty(len(labels), dtype=float)
    for row, neighbor_ids in enumerate(indices):
        _, counts = np.unique(labels[neighbor_ids], return_counts=True)
        probabilities = counts / counts.sum()
        scores[row] = 1.0 / np.square(probabilities).sum()
    return scores


def compute_lisi_table(
    embedding: np.ndarray,
    cell_ids: Sequence[str],
    celltypes: pd.Series,
    batches: pd.Series,
    method: str,
    perplexity: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    metadata = pd.DataFrame(
        {
            "celltype": clean_obs_values(celltypes).to_numpy(),
            "batch": clean_obs_values(batches).to_numpy(),
        }
    )
    source = "harmonypy.compute_lisi"
    try:
        from harmonypy import compute_lisi

        raw = compute_lisi(
            embedding,
            metadata,
            label_colnames=["celltype", "batch"],
            perplexity=min(perplexity, max(2, (len(metadata) - 1) // 3)),
        )
        clisi = raw[:, 0]
        ilisi = raw[:, 1]
    except Exception as exc:
        source = "uniform_knn_fallback"
        logger.warning("Exact Harmony LISI failed (%s); using documented kNN fallback", exc)
        clisi = local_lisi_fallback(embedding, metadata["celltype"], perplexity)
        ilisi = local_lisi_fallback(embedding, metadata["batch"], perplexity)

    n_types = max(1, metadata["celltype"].nunique())
    n_batches = max(1, metadata["batch"].nunique())
    # Match the normalization used in MetaQ's released harmony_integration.py.
    one_minus_clisi = 1.0 - (clisi - 1.0) / n_types
    ilisi_normalized = (ilisi - 1.0) / n_batches
    cell_table = pd.DataFrame(
        {
            "cell_id": np.asarray(cell_ids).astype(str),
            "method": method,
            "celltype": metadata["celltype"],
            "batch": metadata["batch"],
            "cLISI": clisi,
            "iLISI": ilisi,
            "one_minus_cLISI_normalized": one_minus_clisi,
            "iLISI_normalized": ilisi_normalized,
            "implementation": source,
        }
    )
    summary = (
        cell_table.groupby(["method", "celltype", "implementation"], as_index=False)[
            ["one_minus_cLISI_normalized", "iLISI_normalized"]
        ]
        .mean()
        .melt(
            id_vars=["method", "celltype", "implementation"],
            var_name="metric",
            value_name="score",
        )
    )
    return cell_table, summary


def save_embedding_matrix(
    embedding: np.ndarray,
    ids: Sequence[str],
    method: str,
    path: Path,
) -> None:
    frame = pd.DataFrame(embedding, columns=[f"PC{i + 1}" for i in range(embedding.shape[1])])
    frame.insert(0, "item_id", np.asarray(ids).astype(str))
    frame.insert(1, "method", method)
    save_csv(frame, path)


def run_fig4(
    args: argparse.Namespace,
    base: ad.AnnData,
    model: ad.AnnData,
    keys: Mapping[str, str],
    assignments: Mapping[str, pd.Series],
    out_dir: Path,
    target_metacells: int,
    include_original: bool,
    comparison_mode: str = "native",
) -> None:
    safe_mkdir(out_dir)
    celltypes = clean_obs_values(base.obs[keys["celltype"]])
    batches = clean_obs_values(base.obs[keys["batch"]])
    donors = clean_obs_values(base.obs[keys["donor"]])
    conditions = clean_obs_values(base.obs[keys["condition"]])
    truth = celltypes.to_numpy()

    base_hvg = base[:, model.var_names].copy()
    for method, assignment in assignments.items():
        aligned = assignment.reindex(base.obs_names)
        if aligned.isna().any():
            raise ValueError(f"{method}: Fig. 4 assignment is missing evaluation cells")
        logger.info(
            "Fig. 4 %s grid | %s requested=%d realized=%d",
            comparison_mode,
            method,
            target_metacells,
            int(aligned.nunique()),
        )
    if comparison_mode not in {"native", "matched_count"}:
        raise ValueError(f"Unknown Fig. 4 comparison mode: {comparison_mode}")
    protocol_version = (
        MATCHED_FIG4_PROTOCOL_VERSION
        if comparison_mode == "matched_count"
        else FIG4_PROTOCOL_VERSION
    )
    posthoc_kmeans_used = comparison_mode == "matched_count"
    save_json(
        {
            "protocol": protocol_version,
            "comparison_mode": comparison_mode,
            "requested_metacells": target_metacells,
            "evaluation_cells": base.n_obs,
            "method_realized_metacells": {
                method: int(assignment.reindex(base.obs_names).nunique())
                for method, assignment in assignments.items()
            },
            "posthoc_kmeans_used": posthoc_kmeans_used,
            "exact_count_matched": all(
                int(assignment.reindex(base.obs_names).nunique())
                == int(target_metacells)
                for assignment in assignments.values()
            ),
        },
        out_dir / "fig4_protocol.json",
    )
    original_obs = pd.DataFrame(
        {
            "celltype": celltypes.to_numpy(),
            "batch": batches.to_numpy(),
            "donor": donors.to_numpy(),
            "condition": conditions.to_numpy(),
        },
        index=base.obs_names,
    )
    if include_original:
        unintegrated_umap = embedding_umap(
            np.asarray(model.obsm["X_pca"], dtype=np.float32),
            original_obs,
            args.random_seed,
            args.n_neighbors,
        )
        unintegrated_umap["method"] = "Original cells"
        unintegrated_umap["requested_metacells"] = target_metacells
        save_csv(unintegrated_umap, out_dir / "original_unintegrated_umap.csv.gz")

        original_embedding = harmony_integrate(
            np.asarray(model.obsm["X_pca"], dtype=np.float32),
            pd.DataFrame({"batch": batches}, index=base.obs_names),
            "batch",
            args.harmony_iterations,
        )
        save_embedding_matrix(
            original_embedding,
            base.obs_names,
            "Original cells",
            out_dir / "original_integrated_embedding.csv.gz",
        )
        original_umap = embedding_umap(
            original_embedding, original_obs, args.random_seed, args.n_neighbors
        )
        original_umap["method"] = "Original cells"
        original_umap["requested_metacells"] = target_metacells
        save_csv(original_umap, out_dir / "original_recovered_umap.csv.gz")

        clusters, algorithm = cluster_embedding(
            original_embedding,
            args.recovered_resolutions,
            args.random_seed,
            args.n_neighbors,
        )
        metrics, cluster_rows = score_clusters(
            clusters,
            truth,
            method="Original cells",
            representation="integrated_cells",
            algorithm=algorithm,
        )
        metrics["requested_metacells"] = target_metacells
        metrics["realized_metacells"] = base.n_obs
        cluster_rows["requested_metacells"] = target_metacells
        cluster_rows["cell_id"] = np.tile(
            base.obs_names.to_numpy(), len(args.recovered_resolutions)
        )
        save_csv(metrics, out_dir / "original_clustering_metrics.csv")
        save_csv(cluster_rows, out_dir / "original_cluster_assignments.csv.gz")
        lisi_cells, lisi_summary = compute_lisi_table(
            original_embedding,
            base.obs_names,
            celltypes,
            batches,
            "Original cells",
            args.lisi_perplexity,
        )
        lisi_cells["requested_metacells"] = target_metacells
        lisi_cells["realized_metacells"] = base.n_obs
        lisi_summary["requested_metacells"] = target_metacells
        lisi_summary["realized_metacells"] = base.n_obs
        save_csv(lisi_cells, out_dir / "original_lisi_cells.csv.gz")
        save_csv(lisi_summary, out_dir / "original_lisi_summary.csv")

    for method, assignment in assignments.items():
        logger.info("Fig. 4 | starting %s", method)
        method_slug = slugify(method).lower()
        meta, codes = aggregate_metacells(
            base_hvg,
            assignment,
            {"celltype": keys["celltype"], "batch": keys["batch"]},
            method,
        )
        meta_pca = pca_for_metacells(meta, args.n_pcs, args.random_seed)
        meta_harmony = harmony_integrate(
            meta_pca,
            meta.obs,
            "batch",
            args.harmony_iterations,
        )
        save_embedding_matrix(
            meta_harmony,
            meta.obs_names,
            method,
            out_dir / f"{method_slug}_metacell_integrated_embedding.csv.gz",
        )

        meta_umap_obs = meta.obs[["celltype", "batch", "n_cells"]].copy()
        meta_umap = embedding_umap(
            meta_harmony, meta_umap_obs, args.random_seed, max(7, args.n_neighbors // 2)
        )
        meta_umap["method"] = method
        meta_umap["requested_metacells"] = target_metacells
        meta_umap["realized_metacells"] = meta.n_obs
        save_csv(meta_umap, out_dir / f"{method_slug}_metacell_umap.csv.gz")

        meta_clusters, meta_algorithm = cluster_embedding(
            meta_harmony,
            args.metacell_resolutions,
            args.random_seed,
            max(7, args.n_neighbors // 2),
        )
        meta_metrics, meta_cluster_rows = score_clusters(
            meta_clusters,
            truth,
            method=method,
            representation="integrated_metacells_mapped_to_cells",
            algorithm=meta_algorithm,
            expansion_codes=codes,
        )
        meta_metrics["requested_metacells"] = target_metacells
        meta_metrics["realized_metacells"] = meta.n_obs
        meta_cluster_rows["requested_metacells"] = target_metacells
        meta_cluster_rows["cell_id"] = np.tile(
            base.obs_names.to_numpy(), len(args.metacell_resolutions)
        )

        recovered, loss_history = mapping_network(
            meta.X,
            meta_harmony,
            base_hvg.X,
            epochs=args.mapping_epochs,
            batch_size=args.mapping_batch_size,
            device_name=args.device,
            seed=args.random_seed,
        )
        loss_history["method"] = method
        loss_history["requested_metacells"] = target_metacells
        loss_history["realized_metacells"] = meta.n_obs
        save_csv(loss_history, out_dir / f"{method_slug}_mapping_loss.csv")
        save_embedding_matrix(
            recovered,
            base.obs_names,
            method,
            out_dir / f"{method_slug}_recovered_integrated_embedding.csv.gz",
        )

        recovered_obs = original_obs.copy()
        recovered_umap = embedding_umap(
            recovered, recovered_obs, args.random_seed, args.n_neighbors
        )
        recovered_umap["method"] = method
        recovered_umap["requested_metacells"] = target_metacells
        recovered_umap["realized_metacells"] = meta.n_obs
        save_csv(recovered_umap, out_dir / f"{method_slug}_recovered_umap.csv.gz")

        recovered_clusters, recovered_algorithm = cluster_embedding(
            recovered,
            args.recovered_resolutions,
            args.random_seed,
            args.n_neighbors,
        )
        recovered_metrics, recovered_cluster_rows = score_clusters(
            recovered_clusters,
            truth,
            method=method,
            representation="recovered_cells",
            algorithm=recovered_algorithm,
        )
        recovered_metrics["requested_metacells"] = target_metacells
        recovered_metrics["realized_metacells"] = meta.n_obs
        recovered_cluster_rows["requested_metacells"] = target_metacells
        recovered_cluster_rows["cell_id"] = np.tile(
            base.obs_names.to_numpy(), len(args.recovered_resolutions)
        )

        save_csv(
            pd.concat([meta_metrics, recovered_metrics], ignore_index=True),
            out_dir / f"{method_slug}_clustering_metrics.csv",
        )
        save_csv(
            pd.concat([meta_cluster_rows, recovered_cluster_rows], ignore_index=True),
            out_dir / f"{method_slug}_cluster_assignments.csv.gz",
        )
        lisi_cells, lisi_summary = compute_lisi_table(
            recovered,
            base.obs_names,
            celltypes,
            batches,
            method,
            args.lisi_perplexity,
        )
        lisi_cells["requested_metacells"] = target_metacells
        lisi_cells["realized_metacells"] = meta.n_obs
        lisi_summary["requested_metacells"] = target_metacells
        lisi_summary["realized_metacells"] = meta.n_obs
        save_csv(lisi_cells, out_dir / f"{method_slug}_lisi_cells.csv.gz")
        save_csv(lisi_summary, out_dir / f"{method_slug}_lisi_summary.csv")
        logger.info("Fig. 4 | finished %s", method)


def fig4_target_is_complete(
    out_dir: Path,
    methods: Sequence[str],
    include_original: bool,
    requested_metacells: Optional[int] = None,
    expected_protocol: str = FIG4_PROTOCOL_VERSION,
    expected_posthoc_kmeans: bool = False,
) -> bool:
    """Return True only when every expensive checkpoint for a target exists."""
    protocol_path = out_dir / "fig4_protocol.json"
    if not protocol_path.is_file():
        return False
    try:
        protocol = json.loads(protocol_path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    if (
        protocol.get("protocol") != expected_protocol
        or protocol.get("posthoc_kmeans_used") is not expected_posthoc_kmeans
        or (
            expected_protocol == MATCHED_FIG4_PROTOCOL_VERSION
            and protocol.get("exact_count_matched") is not True
        )
        or (
            requested_metacells is not None
            and int(protocol.get("requested_metacells", -1)) != requested_metacells
        )
        or set(protocol.get("method_realized_metacells", {})) != set(methods)
    ):
        return False
    method_suffixes = [
        "metacell_integrated_embedding.csv.gz",
        "metacell_umap.csv.gz",
        "mapping_loss.csv",
        "recovered_integrated_embedding.csv.gz",
        "recovered_umap.csv.gz",
        "clustering_metrics.csv",
        "cluster_assignments.csv.gz",
        "lisi_cells.csv.gz",
        "lisi_summary.csv",
    ]
    required = [protocol_path]
    required.extend(
        [
            out_dir / f"{slugify(method).lower()}_{suffix}"
            for method in methods
            for suffix in method_suffixes
        ]
    )
    if include_original:
        required.extend(
            [
                out_dir / "original_unintegrated_umap.csv.gz",
                out_dir / "original_integrated_embedding.csv.gz",
                out_dir / "original_recovered_umap.csv.gz",
                out_dir / "original_clustering_metrics.csv",
                out_dir / "original_cluster_assignments.csv.gz",
                out_dir / "original_lisi_cells.csv.gz",
                out_dir / "original_lisi_summary.csv",
            ]
        )
    return bool(required) and all(path.is_file() for path in required)


# =========================================================
# CAMP partition construction and Fig. 5 DE preservation
# =========================================================
def compute_coreset_q(X) -> np.ndarray:
    if sparse.issparse(X):
        X = X.tocsr(copy=False)
        mean = np.asarray(X.mean(axis=0)).ravel()
        d_sq = (
            np.asarray(X.power(2).sum(axis=1)).ravel()
            + float(mean @ mean)
            - 2.0 * np.asarray(X @ mean).ravel()
        )
    else:
        X = np.asarray(X, dtype=np.float64)
        mean = X.mean(axis=0)
        d_sq = np.square(X - mean).sum(axis=1)
    d_sq = np.maximum(d_sq, 0.0)
    if float(d_sq.sum()) == 0.0:
        return np.full(X.shape[0], 1.0 / X.shape[0])
    q = 0.5 / X.shape[0] + 0.5 * d_sq / d_sq.sum()
    return q / q.sum()


def adaptive_kernel(X: np.ndarray, n_neighbors: int = 15) -> csr_matrix:
    n_neighbors = min(n_neighbors, X.shape[0])
    nn = NearestNeighbors(n_neighbors=n_neighbors).fit(X)
    distances, indices = nn.kneighbors(X)
    rows = np.repeat(np.arange(X.shape[0]), n_neighbors)
    matrix = csr_matrix(
        (distances.ravel(), (rows, indices.ravel())),
        shape=(X.shape[0], X.shape[0]),
    )
    matrix = matrix.maximum(matrix.T)
    sigma = np.maximum(distances[:, -1], 1e-8)
    coo = matrix.tocoo()
    values = np.exp(-np.square(coo.data) / (sigma[coo.row] * sigma[coo.col]))
    return csr_matrix((values, (coo.row, coo.col)), shape=matrix.shape)


def assign_camp_group_details(
    X_pca: np.ndarray,
    variant: str,
    n_metacells: int,
    rng: np.random.RandomState,
) -> Tuple[np.ndarray, np.ndarray]:
    """Literal CAMP1-4 PBMC assignment core from the non-on-the-fly scripts."""
    n_metacells = min(max(1, n_metacells), X_pca.shape[0])
    variant = variant.lower()

    if X_pca.shape[0] <= 2 or n_metacells == 1:
        q = compute_coreset_q(X_pca)
        seed_indices = np.asarray(
            rng.choice(X_pca.shape[0], n_metacells, replace=False, p=q), dtype=int
        )
        return np.zeros(X_pca.shape[0], dtype=int), seed_indices

    if variant in {"camp1", "camp4"}:
        q = compute_coreset_q(X_pca)
    elif variant == "camp2":
        logger.info("CAMP2: building the full PBMC cosine-similarity matrix")
        X_unit = normalize(X_pca, axis=1, copy=True)
        similarity = X_unit.dot(X_unit.T)
        q = compute_coreset_q(similarity)
    elif variant == "camp3":
        kernel = adaptive_kernel(X_pca, n_neighbors=15)
        q = compute_coreset_q(kernel)
    else:
        raise ValueError(f"Unsupported CAMP variant: {variant}")

    seed_indices = np.asarray(
        rng.choice(X_pca.shape[0], n_metacells, replace=False, p=q), dtype=int
    )
    if variant == "camp1":
        model = KMeans(
            n_clusters=n_metacells,
            init=X_pca[seed_indices],
            n_init=1,
            max_iter=10,
            algorithm="lloyd",
        )
        return model.fit_predict(X_pca), seed_indices

    if variant == "camp2":
        labels = np.argmax(similarity[:, seed_indices], axis=1)
        return np.asarray(labels).ravel(), seed_indices

    if variant == "camp3":
        block = kernel[:, seed_indices].toarray()
        labels = np.argmax(block, axis=1)
        return np.asarray(labels).ravel(), seed_indices

    # CAMP4: exact fixed-archetype SEACells kernel used by the released PBMC script.
    build_graph = load_seacells_build_graph()
    logger.info("Using SEACells graph code: %s", Path(build_graph.__file__).resolve())
    subset = ad.AnnData(X=np.zeros((X_pca.shape[0], 1), dtype=np.float32))
    subset.obs_names = pd.Index([f"cell_{i}" for i in range(X_pca.shape[0])])
    subset.obsm["X_pca"] = X_pca
    seacells_cores = max(
        1,
        int(os.environ.get("SLURM_CPUS_PER_TASK", os.environ.get("OMP_NUM_THREADS", "1"))),
    )
    try:
        graph = build_graph.SEACellGraph(
            subset,
            "X_pca",
            n_cores=seacells_cores,
            verbose=False,
        )
    except TypeError:
        graph = build_graph.SEACellGraph(subset, "X_pca", verbose=False)
    n_neighbors = min(30, max(2, subset.n_obs - 1))
    rbf_parameters = inspect.signature(graph.rbf).parameters
    if "n_neighbors" in rbf_parameters:
        kernel = graph.rbf(n_neighbors=n_neighbors, graph_construction="union")
    elif "k" in rbf_parameters:
        kernel = graph.rbf(k=n_neighbors, graph_construction="union")
    else:
        graph.n_neighbors = n_neighbors
        kernel = graph.rbf(graph_construction="union")
    block = kernel.tocsr()[:, seed_indices].toarray()
    labels = np.argmax(block, axis=1)
    return np.asarray(labels).ravel(), seed_indices


def assign_camp_group(
    X_pca: np.ndarray,
    variant: str,
    n_metacells: int,
    rng: np.random.RandomState,
) -> np.ndarray:
    labels, _ = assign_camp_group_details(X_pca, variant, n_metacells, rng)
    return labels


def build_or_load_fig4_partitions(
    model: ad.AnnData,
    variants: Sequence[str],
    gamma: int,
    seed: int,
    partition_dir: Path,
    force: bool,
) -> Dict[str, pd.Series]:
    """Generate CAMP PBMC partitions instead of consuming external CSVs."""
    safe_mkdir(partition_dir)
    assignments: Dict[str, pd.Series] = {}
    summary_rows = []
    X_pca = np.asarray(model.obsm["X_pca"], dtype=np.float32)
    target_metacells = max(1, model.n_obs // gamma)

    for variant_name in variants:
        variant = variant_name.lower()
        method = display_name(variant)
        output_path = partition_dir / f"{variant}_generated_partitions.csv"
        if output_path.exists() and not force:
            frame = pd.read_csv(output_path, index_col=0)
            frame.index = frame.index.astype(str)
            if str(gamma) not in frame.columns:
                raise KeyError(f"Cached partition {output_path} lacks gamma {gamma}")
            assignment = frame.loc[model.obs_names, str(gamma)].astype(str)
            logger.info("Using generated partition checkpoint: %s", output_path)
        else:
            logger.info(
                "Constructing %s partition with non-on-the-fly PBMC code | gamma=%d | m=%d",
                method,
                gamma,
                target_metacells,
            )
            rng = np.random.RandomState(seed)
            labels, seed_indices = assign_camp_group_details(
                X_pca,
                variant,
                target_metacells,
                rng,
            )
            assignment = pd.Series(
                [f"seed{label}-{gamma}-allcells" for label in labels],
                index=model.obs_names,
                dtype="string",
            )
            seed_flags = np.zeros(model.n_obs, dtype=bool)
            seed_flags[seed_indices] = True
            frame = pd.DataFrame(
                {
                    str(gamma): assignment.to_numpy(),
                    f"{gamma}_is_seed": seed_flags,
                },
                index=model.obs_names,
            )
            save_csv(frame, output_path, index=True)
        assignment.index = model.obs_names
        assignments[method] = assignment
        n_realized = assignment.nunique()
        summary_rows.append(
            {
                "method": method,
                "construction": "CAMP/without_on_the_fly_for_pbmc_data",
                "gamma": gamma,
                "target_metacells": target_metacells,
                "realized_metacells": n_realized,
                "effective_reduction_rate": model.n_obs / n_realized,
                "partition_checkpoint": str(output_path),
            }
        )

    save_csv(pd.DataFrame(summary_rows), partition_dir / "partition_summary.csv")
    return assignments


def load_released_pbmc_baseline_partitions(
    model: ad.AnnData,
    baseline_root: Path,
    gamma: int,
    partition_dir: Path,
) -> Tuple[Dict[str, pd.Series], pd.Index]:
    """Load the native PBMC memberships shipped with the CAMP work tree.

    MetaCell1 and SuperCell are R-native in the released repository. Reusing
    their released membership CSVs preserves those native runs and avoids
    replacing them with Python approximations. Evaluation is restricted to the
    ordered intersection of cells with non-missing membership in every method;
    this preserves native outlier handling and gives all methods identical
    cell coverage.
    """
    if gamma != 20:
        raise ValueError(
            "Released PBMC baseline memberships are matched at gamma=20. "
            "Use --fig4-gamma 20 when baselines are enabled."
        )

    output_dir = partition_dir / "released_pbmc_baselines"
    safe_mkdir(output_dir)
    native_assignments: Dict[str, pd.Series] = {}
    summary_rows = []
    common_mask = np.ones(model.n_obs, dtype=bool)

    for method, spec in RELEASED_BASELINES.items():
        source_path = baseline_root / str(spec["relative_path"])
        if not source_path.is_file():
            raise FileNotFoundError(
                f"Missing released {method} PBMC membership: {source_path}. "
                "Upload the complete SEACells folder beside the Python script."
            )
        frame = pd.read_csv(source_path, index_col=0, low_memory=False)
        frame.index = frame.index.astype(str)
        if frame.index.has_duplicates:
            raise ValueError(f"{method}: duplicate cell IDs in {source_path}")
        column = str(spec["column"])
        frame.columns = frame.columns.astype(str)
        if column not in frame.columns:
            raise KeyError(
                f"{method}: expected column '{column}' in {source_path}; "
                f"found {frame.columns.tolist()}"
            )

        native = frame[column].copy()
        valid = native.notna() & (native.astype(str).str.strip() != "")
        native = native.loc[valid].astype(str)
        native_assignments[method] = native
        method_valid_mask = model.obs_names.isin(native.index)
        common_mask &= method_valid_mask
        summary_rows.append(
            {
                "method": method,
                "construction": "released_native_PBMC_membership",
                "source_file": str(source_path.resolve()),
                "source_column": column,
                "native_parameter": spec["native_parameter"],
                "full_preprocessed_cells": model.n_obs,
                "source_rows": len(frame),
                "valid_membership_cells_in_preprocessed_data": int(method_valid_mask.sum()),
                "missing_or_native_unassigned_cells": int((~method_valid_mask).sum()),
            }
        )

    common_ids = model.obs_names[common_mask]
    if len(common_ids) < 2:
        raise ValueError("Released PBMC memberships have no usable common cell universe")
    assignments: Dict[str, pd.Series] = {}
    for row, (method, native) in zip(summary_rows, native_assignments.items()):
        assignment = native.reindex(common_ids)
        if assignment.isna().any():
            raise RuntimeError(f"{method}: internal error aligning common PBMC cells")
        assignment = pd.Series(
            [f"{slugify(method).lower()}|native={value}" for value in assignment],
            index=common_ids,
            dtype="string",
        )
        assignments[method] = assignment.astype(str)
        checkpoint_path = output_dir / f"{slugify(method).lower()}_gamma20_common.csv.gz"
        save_csv(
            pd.DataFrame(
                {"cell_id": assignment.index.astype(str), "metacell": assignment.to_numpy()}
            ),
            checkpoint_path,
        )
        n_metacells = int(assignment.nunique())
        row.update(
            {
                "common_evaluation_cells": len(common_ids),
                "valid_cells_excluded_for_other_methods": int(
                    len(native.index.intersection(model.obs_names)) - len(common_ids)
                ),
                "realized_metacells_on_common_cells": n_metacells,
                "effective_reduction_rate_on_common_cells": len(common_ids) / n_metacells,
                "checkpoint": str(checkpoint_path),
            }
        )
        logger.info(
            "Loaded released %s membership | column=%s common_cells=%d metacells=%d",
            method,
            RELEASED_BASELINES[method]["column"],
            len(common_ids),
            n_metacells,
        )

    save_csv(
        pd.DataFrame({"cell_id": common_ids.astype(str)}),
        output_dir / "common_evaluation_cell_ids.csv.gz",
    )
    save_csv(pd.DataFrame(summary_rows), output_dir / "baseline_partition_summary.csv")
    logger.info(
        "All-method evaluation uses %d/%d preprocessed PBMC cells with native "
        "membership in every baseline",
        len(common_ids),
        model.n_obs,
    )
    return assignments, common_ids


def stable_seed(base_seed: int, *parts: object) -> int:
    """Return a process-independent uint32 seed for a named subproblem."""
    payload = "|".join([str(base_seed), *[str(part) for part in parts]])
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="little", signed=False)


def _valid_native_membership(values: pd.Series) -> pd.Series:
    """Identify non-empty native memberships without converting NaN to text."""
    as_text = values.astype("string")
    return as_text.notna() & (as_text.str.strip() != "")


def load_released_pbmc_native_grid(
    model: ad.AnnData,
    baseline_root: Path,
    targets: Sequence[int],
    partition_dir: Path,
) -> Tuple[Dict[int, Dict[str, pd.Series]], pd.Index, pd.DataFrame]:
    """Select the closest released native baseline partition at each target.

    The common evaluation universe is fixed before resolution selection: a cell
    must have a valid assignment in every released resolution column for every
    baseline. Counts are then measured on that same universe, and distinct
    columns minimizing total distance to the requested grid are selected. No
    membership is merged, split, or reclustered.
    """
    output_dir = partition_dir / "native_resolution_grid" / "released_baselines"
    safe_mkdir(output_dir)
    frames: Dict[str, pd.DataFrame] = {}
    source_paths: Dict[str, Path] = {}
    common_mask = np.ones(model.n_obs, dtype=bool)

    for method, spec in RELEASED_BASELINES.items():
        source_path = baseline_root / str(spec["relative_path"])
        if not source_path.is_file():
            raise FileNotFoundError(
                f"Missing released {method} membership: {source_path}. "
                "Upload the complete SEACells folder beside the Python script."
            )
        frame = pd.read_csv(source_path, index_col=0, low_memory=False)
        frame.index = frame.index.astype(str)
        frame.columns = frame.columns.astype(str)
        if frame.index.has_duplicates:
            raise ValueError(f"{method}: duplicate cell IDs in {source_path}")
        numeric_columns = [
            column for column in frame.columns if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", column)
        ]
        if not numeric_columns:
            raise ValueError(f"{method}: no numeric native resolution columns in {source_path}")
        frame = frame[numeric_columns]
        frames[method] = frame
        source_paths[method] = source_path.resolve()

        in_source = model.obs_names.isin(frame.index)
        aligned = frame.reindex(model.obs_names)
        valid_all_columns = pd.DataFrame(
            {column: _valid_native_membership(aligned[column]) for column in frame.columns},
            index=model.obs_names,
        ).all(axis=1).to_numpy()
        common_mask &= in_source & valid_all_columns

    common_ids = model.obs_names[common_mask]
    if len(common_ids) < 2:
        raise ValueError("Released native grids have no usable common cell universe")

    assignments_by_target: Dict[int, Dict[str, pd.Series]] = {
        int(target): {} for target in targets
    }
    summary_rows: List[Dict[str, object]] = []
    for method, frame in frames.items():
        aligned = frame.reindex(common_ids)
        realized_by_column = {
            column: int(aligned[column].astype(str).nunique()) for column in aligned.columns
        }
        requested_values = [int(target) for target in targets]
        if len(aligned.columns) < len(requested_values):
            raise ValueError(
                f"{method}: {len(aligned.columns)} released native resolutions cannot "
                f"supply {len(requested_values)} distinct sensitivity points"
            )
        cost = np.asarray(
            [
                [abs(realized_by_column[column] - requested) for column in aligned.columns]
                for requested in requested_values
            ],
            dtype=float,
        )
        # A one-to-one minimum-cost assignment prevents a sparse baseline grid
        # from silently reusing one native partition at multiple requested points.
        requested_rows, selected_columns = linear_sum_assignment(cost)
        selected_by_requested = {
            requested_values[row]: aligned.columns[column]
            for row, column in zip(requested_rows, selected_columns)
        }
        for requested in requested_values:
            selected_column = selected_by_requested[requested]
            raw = aligned[selected_column].astype(str)
            assignment = pd.Series(
                [f"{slugify(method).lower()}|native={value}" for value in raw],
                index=common_ids,
                dtype="string",
            ).astype(str)
            assignments_by_target[requested][method] = assignment
            realized = int(assignment.nunique())
            target_dir = output_dir / f"requested_m{requested}"
            safe_mkdir(target_dir)
            checkpoint = target_dir / f"{slugify(method).lower()}_native_assignment.csv.gz"
            save_csv(
                pd.DataFrame(
                    {"cell_id": common_ids.astype(str), "metacell": assignment.to_numpy()}
                ),
                checkpoint,
            )
            summary_rows.append(
                {
                    "protocol": FIG4_PROTOCOL_VERSION,
                    "method": method,
                    "construction": "released_native_partition_no_posthoc_clustering",
                    "requested_metacells": requested,
                    "source_parameter_name": RELEASED_BASELINES[method]["parameter_name"],
                    "source_column": selected_column,
                    "realized_metacells": realized,
                    "absolute_target_difference": abs(realized - requested),
                    "relative_target_difference": abs(realized - requested) / requested,
                    "evaluation_cells": len(common_ids),
                    "effective_reduction_rate": len(common_ids) / realized,
                    "source_file": str(source_paths[method]),
                    "checkpoint": str(checkpoint),
                    "posthoc_kmeans_used": False,
                }
            )
            logger.info(
                "Native baseline grid | %s requested=%d column=%s realized=%d",
                method,
                requested,
                selected_column,
                realized,
            )

    summary = pd.DataFrame(summary_rows)
    save_csv(summary, output_dir / "native_resolution_selection.csv")
    save_csv(
        pd.DataFrame({"cell_id": common_ids.astype(str)}),
        output_dir / "common_evaluation_cell_ids.csv.gz",
    )
    save_json(
        {
            "protocol": FIG4_PROTOCOL_VERSION,
            "selection_rule": (
                "one-to-one minimum-total-distance assignment of requested counts to "
                "distinct released native resolution columns, measured on the fixed "
                "all-method/all-resolution common cell universe"
            ),
            "posthoc_kmeans_used": False,
            "evaluation_cells": len(common_ids),
            "requested_metacells": [int(target) for target in targets],
        },
        output_dir / "native_resolution_protocol.json",
    )
    return assignments_by_target, common_ids, summary


def load_released_pbmc_merge_sources(
    model: ad.AnnData,
    baseline_root: Path,
    common_ids: pd.Index,
    targets: Sequence[int],
    partition_dir: Path,
) -> Tuple[Dict[int, Dict[str, pd.Series]], pd.DataFrame]:
    """Load the closest merge-feasible released source at every target.

    Exact-count Fig. 4 comparisons must be merge-only: splitting a method's
    native metacells would inject new cell-level clustering into that method.
    For each target and baseline, we therefore choose the released partition
    having the smallest realized count that is still at least the target. This
    minimizes post-processing, then merges only intact native fragments by the
    common weighted-centroid rule.
    """
    requested_targets = [int(target) for target in targets]
    output_dir = partition_dir / "matched_count_grid" / "merge_sources"
    safe_mkdir(output_dir)
    assignments_by_target: Dict[int, Dict[str, pd.Series]] = {
        target: {} for target in requested_targets
    }
    rows: List[Dict[str, object]] = []
    for method, spec in RELEASED_BASELINES.items():
        source_path = baseline_root / str(spec["relative_path"])
        frame = pd.read_csv(source_path, index_col=0, low_memory=False)
        frame.index = frame.index.astype(str)
        frame.columns = frame.columns.astype(str)
        numeric_columns = [
            column
            for column in frame.columns
            if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", column)
        ]
        aligned = frame.reindex(common_ids)[numeric_columns]
        valid = pd.DataFrame(
            {
                column: _valid_native_membership(aligned[column])
                for column in numeric_columns
            },
            index=common_ids,
        )
        if not bool(valid.all().all()):
            raise ValueError(
                f"{method}: high-resolution source is incomplete on the common cell universe"
            )
        realized = {
            column: int(aligned[column].astype(str).nunique())
            for column in numeric_columns
        }
        for target in requested_targets:
            feasible_columns = [
                column for column in numeric_columns if realized[column] >= target
            ]
            if not feasible_columns:
                maximum = max(realized.values())
                raise ValueError(
                    f"{method}: finest released source has {maximum} metacells, "
                    f"below the requested matched target {target}"
                )
            selected_column = min(
                feasible_columns,
                key=lambda column: (realized[column] - target, float(column)),
            )
            source_count = realized[selected_column]
            raw = aligned[selected_column].astype(str)
            assignment = pd.Series(
                [f"{slugify(method).lower()}|native={value}" for value in raw],
                index=common_ids,
                dtype="string",
            ).astype(str)
            assignments_by_target[target][method] = assignment
            target_dir = output_dir / f"m{target}"
            safe_mkdir(target_dir)
            checkpoint = (
                target_dir / f"{slugify(method).lower()}_source_assignment.csv.gz"
            )
            save_csv(
                pd.DataFrame(
                    {
                        "cell_id": common_ids.astype(str),
                        "metacell": assignment.to_numpy(),
                    }
                ),
                checkpoint,
            )
            rows.append(
                {
                    "protocol": MATCHED_FIG4_PROTOCOL_VERSION,
                    "method": method,
                    "target_metacells": target,
                    "selection_rule": "closest_released_count_at_or_above_target",
                    "source_parameter_name": str(spec["parameter_name"]),
                    "source_column": selected_column,
                    "source_metacells": source_count,
                    "source_minus_target": source_count - target,
                    "evaluation_cells": len(common_ids),
                    "source_file": str(source_path.resolve()),
                    "checkpoint": str(checkpoint),
                }
            )
            logger.info(
                "Matched Fig. 4 source | %s target_m=%d column=%s native_m=%d",
                method,
                target,
                selected_column,
                source_count,
            )
    summary = pd.DataFrame(rows)
    save_csv(summary, output_dir / "merge_source_summary.csv")
    return assignments_by_target, summary


def build_or_load_camp_native_grid(
    model: ad.AnnData,
    variants: Sequence[str],
    targets: Sequence[int],
    seed: int,
    partition_dir: Path,
    force: bool,
) -> Tuple[Dict[int, Dict[str, pd.Series]], pd.DataFrame]:
    """Run each CAMP variant directly at every requested metacell count."""
    root = partition_dir / "native_resolution_grid" / "camp"
    safe_mkdir(root)
    X_pca = np.asarray(model.obsm["X_pca"], dtype=np.float32)
    assignments_by_target: Dict[int, Dict[str, pd.Series]] = {
        int(target): {} for target in targets
    }
    summary_rows: List[Dict[str, object]] = []

    for requested in [int(target) for target in targets]:
        if not 2 <= requested <= model.n_obs:
            raise ValueError(
                f"Requested CAMP metacell count {requested} is infeasible for {model.n_obs} cells"
            )
        target_dir = root / f"requested_m{requested}"
        safe_mkdir(target_dir)
        for variant_name in variants:
            variant = variant_name.lower()
            method = display_name(variant)
            checkpoint = target_dir / f"{variant}_native_assignment.csv.gz"
            metadata_path = target_dir / f"{variant}_native_assignment.json"
            use_cache = checkpoint.is_file() and metadata_path.is_file() and not force
            if use_cache:
                metadata = json.loads(metadata_path.read_text())
                use_cache = (
                    metadata.get("protocol") == FIG4_PROTOCOL_VERSION
                    and int(metadata.get("requested_metacells", -1)) == requested
                    and int(metadata.get("evaluation_cells", -1)) == model.n_obs
                )
            if use_cache:
                frame = pd.read_csv(checkpoint)
                assignment = pd.Series(
                    frame["metacell"].astype(str).to_numpy(),
                    index=frame["cell_id"].astype(str),
                ).reindex(model.obs_names)
                if assignment.isna().any():
                    raise ValueError(f"Cached CAMP assignment cannot be aligned: {checkpoint}")
                logger.info("Using CAMP native-grid checkpoint: %s", checkpoint)
            else:
                logger.info(
                    "Constructing native %s partition | requested metacells=%d",
                    method,
                    requested,
                )
                # Reset to the same user seed for each method at a given target,
                # matching the released CAMP experiment convention.
                rng = np.random.RandomState(seed)
                labels, seed_indices = assign_camp_group_details(
                    X_pca,
                    variant,
                    requested,
                    rng,
                )
                assignment = pd.Series(
                    [f"{variant}|native={label}|requested={requested}" for label in labels],
                    index=model.obs_names,
                    dtype="string",
                ).astype(str)
                seed_flags = np.zeros(model.n_obs, dtype=bool)
                seed_flags[np.asarray(seed_indices, dtype=int)] = True
                save_csv(
                    pd.DataFrame(
                        {
                            "cell_id": model.obs_names.astype(str),
                            "metacell": assignment.to_numpy(),
                            "is_seed": seed_flags,
                        }
                    ),
                    checkpoint,
                )
                save_json(
                    {
                        "protocol": FIG4_PROTOCOL_VERSION,
                        "method": method,
                        "variant": variant,
                        "construction": "CAMP_native_direct_target",
                        "requested_metacells": requested,
                        "realized_metacells": int(assignment.nunique()),
                        "evaluation_cells": model.n_obs,
                        "random_seed": seed,
                        "posthoc_kmeans_used": False,
                    },
                    metadata_path,
                )
            assignment.index = model.obs_names
            assignments_by_target[requested][method] = assignment.astype(str)
            realized = int(assignment.nunique())
            summary_rows.append(
                {
                    "protocol": FIG4_PROTOCOL_VERSION,
                    "method": method,
                    "construction": "CAMP_native_direct_target",
                    "requested_metacells": requested,
                    "source_parameter_name": "target_n_metacells",
                    "source_column": requested,
                    "realized_metacells": realized,
                    "absolute_target_difference": abs(realized - requested),
                    "relative_target_difference": abs(realized - requested) / requested,
                    "evaluation_cells": model.n_obs,
                    "effective_reduction_rate": model.n_obs / realized,
                    "source_file": str(Path(__file__).resolve()),
                    "checkpoint": str(checkpoint),
                    "posthoc_kmeans_used": False,
                }
            )

    summary = pd.DataFrame(summary_rows)
    save_csv(summary, root / "camp_native_resolution_summary.csv")
    return assignments_by_target, summary


def partition_signature(assignment: pd.Series) -> str:
    """Hash an aligned assignment so matched-count checkpoints are auditable."""
    frame = pd.DataFrame(
        {
            "cell_id": assignment.index.astype(str),
            "metacell": assignment.astype(str).to_numpy(),
        }
    )
    hashed = pd.util.hash_pandas_object(frame, index=False).to_numpy(dtype=np.uint64)
    return hashlib.sha256(hashed.tobytes()).hexdigest()


def coarsen_native_partition_to_exact_count(
    model: ad.AnnData,
    assignment: pd.Series,
    method: str,
    target_metacells: int,
    seed: int,
) -> Tuple[pd.Series, pd.DataFrame]:
    """Merge intact native fragments to one exact, shared Fig. 4 budget.

    Native metacells are represented by size-weighted centroids in the common
    PBMC PCA space. Weighted k-means merges those fragments to the requested
    count; it never splits a native metacell or moves individual cells between
    native fragments. The same transformation and random seed are used for
    every method.
    """
    assignment = assignment.reindex(model.obs_names)
    if assignment.isna().any():
        raise ValueError(f"{method}: source assignment is missing evaluation cells")
    if target_metacells < 2:
        raise ValueError("--fig4-target-metacells values must be at least 2")

    source_ids, source_codes = np.unique(
        assignment.astype(str).to_numpy(), return_inverse=True
    )
    n_source = len(source_ids)
    if target_metacells > n_source:
        raise ValueError(
            f"{method}: cannot obtain {target_metacells} metacells by merge-only "
            f"coarsening because the native source has {n_source}"
        )

    sizes = np.bincount(source_codes, minlength=n_source).astype(np.float64)
    X_pca = np.asarray(model.obsm["X_pca"], dtype=np.float32)
    membership = csr_matrix(
        (
            1.0 / sizes[source_codes],
            (source_codes, np.arange(model.n_obs)),
        ),
        shape=(n_source, model.n_obs),
        dtype=np.float64,
    )
    centroids = np.asarray(membership @ X_pca, dtype=np.float64)

    if target_metacells == n_source:
        matched_source_labels = np.arange(n_source, dtype=int)
    else:
        coarsener = KMeans(
            n_clusters=target_metacells,
            init="k-means++",
            n_init=3,
            max_iter=200,
            algorithm="lloyd",
            random_state=stable_seed(
                seed, MATCHED_FIG4_PROTOCOL_VERSION, method, target_metacells
            ),
        )
        coarsener.fit(centroids, sample_weight=sizes)
        matched_source_labels = np.asarray(coarsener.labels_, dtype=int)

        # Exact duplicated centroids can leave an empty k-means label. Restore
        # the requested count by peeling intact native fragments from coarse
        # groups; cells inside a native metacell are still never split.
        realized_labels = np.unique(matched_source_labels)
        next_label = int(realized_labels.max()) + 1
        while len(realized_labels) < target_metacells:
            group_counts = pd.Series(matched_source_labels).value_counts()
            splittable = group_counts[group_counts > 1]
            if splittable.empty:
                raise RuntimeError(
                    f"{method}: could not construct exactly {target_metacells} groups"
                )
            group = int(splittable.index[0])
            candidates = np.flatnonzero(matched_source_labels == group)
            fragment = int(candidates[np.argmin(sizes[candidates])])
            matched_source_labels[fragment] = next_label
            next_label += 1
            realized_labels = np.unique(matched_source_labels)

    # Canonicalize arbitrary k-means labels by the first native fragment in
    # each group, yielding stable human-readable IDs.
    label_order = sorted(
        np.unique(matched_source_labels),
        key=lambda label: int(np.flatnonzero(matched_source_labels == label)[0]),
    )
    canonical = {int(label): position for position, label in enumerate(label_order)}
    matched_source_labels = np.asarray(
        [canonical[int(label)] for label in matched_source_labels], dtype=int
    )
    cell_labels = matched_source_labels[source_codes]
    method_slug = slugify(method).lower()
    matched = pd.Series(
        [f"{method_slug}|matched_m{target_metacells}|{label}" for label in cell_labels],
        index=model.obs_names,
        dtype="string",
    )
    if matched.nunique() != target_metacells:
        raise RuntimeError(
            f"{method}: exact-count validation failed for m={target_metacells}"
        )

    fragment_map = pd.DataFrame(
        {
            "method": method,
            "target_metacells": target_metacells,
            "source_metacell": source_ids.astype(str),
            "source_n_cells": sizes.astype(int),
            "matched_metacell": [
                f"{method_slug}|matched_m{target_metacells}|{label}"
                for label in matched_source_labels
            ],
        }
    )
    return matched.astype(str), fragment_map


def build_matched_fig4_assignments(
    model: ad.AnnData,
    source_assignments_by_target: Mapping[int, Mapping[str, pd.Series]],
    targets: Sequence[int],
    seed: int,
    output_dir: Path,
) -> Dict[int, Dict[str, pd.Series]]:
    """Build and validate exact matched-count assignments for all methods."""
    safe_mkdir(output_dir)
    targets = [int(target) for target in targets]
    if not targets:
        raise ValueError("At least one --fig4-target-metacells value is required")
    if len(set(targets)) != len(targets):
        raise ValueError("--fig4-target-metacells contains duplicate values")

    matched_by_target: Dict[int, Dict[str, pd.Series]] = {}
    summary_rows = []
    for target in targets:
        if target not in source_assignments_by_target:
            raise KeyError(f"Missing Fig. 4 source assignments for target m={target}")
        target_dir = output_dir / f"m{target}"
        safe_mkdir(target_dir)
        matched_by_target[target] = {}
        for method, source in source_assignments_by_target[target].items():
            aligned_source = source.reindex(model.obs_names)
            source_count = int(aligned_source.nunique())
            matched, fragment_map = coarsen_native_partition_to_exact_count(
                model=model,
                assignment=aligned_source,
                method=method,
                target_metacells=target,
                seed=seed,
            )
            matched_by_target[target][method] = matched
            method_slug = slugify(method).lower()
            assignment_path = target_dir / f"{method_slug}_matched_assignment.csv.gz"
            mapping_path = target_dir / f"{method_slug}_native_fragment_map.csv.gz"
            save_csv(
                pd.DataFrame(
                    {
                        "cell_id": matched.index.astype(str),
                        "metacell": matched.to_numpy(),
                    }
                ),
                assignment_path,
            )
            save_csv(fragment_map, mapping_path)
            summary_rows.append(
                {
                    "protocol": MATCHED_FIG4_PROTOCOL_VERSION,
                    "method": method,
                    "evaluation_cells": model.n_obs,
                    "target_metacells": target,
                    "source_metacells": source_count,
                    "realized_metacells": int(matched.nunique()),
                    "exact_target_pass": int(matched.nunique()) == target,
                    "source_reduction_rate": model.n_obs / source_count,
                    "matched_reduction_rate": model.n_obs / target,
                    "operation": "merge_only_weighted_kmeans_on_native_PCA_centroids",
                    "native_fragments_split": 0,
                    "source_partition_sha256": partition_signature(aligned_source.astype(str)),
                    "assignment_checkpoint": str(assignment_path),
                    "fragment_map_checkpoint": str(mapping_path),
                }
            )
            logger.info(
                "Matched Fig. 4 partition | method=%s source_m=%d target_m=%d cells=%d",
                method,
                source_count,
                target,
                model.n_obs,
            )

    summary = pd.DataFrame(summary_rows)
    if not summary["exact_target_pass"].all():
        raise RuntimeError("At least one Fig. 4 assignment missed its exact target")
    counts_per_target = summary.groupby("target_metacells")["realized_metacells"].nunique()
    if not (counts_per_target == 1).all():
        raise RuntimeError("Fig. 4 methods do not have identical realized counts")
    save_csv(summary, output_dir / "fig4_matched_count_validation.csv")
    save_json(
        {
            "protocol": MATCHED_FIG4_PROTOCOL_VERSION,
            "targets": targets,
            "evaluation_cells": model.n_obs,
            "method_count": int(summary["method"].nunique()),
            "contract": (
                "same cells; exact same metacell count at each target; closest "
                "merge-feasible released source per baseline; CAMP direct target; "
                "native fragments merged but never split"
            ),
        },
        output_dir / "fig4_matched_count_protocol.json",
    )
    return matched_by_target


def allocate_stratum_metacell_targets(
    model: ad.AnnData,
    keys: Mapping[str, str],
    reduction_rate: float,
) -> pd.DataFrame:
    """Allocate one shared, exact metacell budget across all DE strata.

    The total budget is round(N / reduction_rate). Every non-empty stratum is
    guaranteed at least one profile, and remaining profiles are assigned by
    largest-remainder apportionment. The returned table is method-independent
    and is therefore the contract every method must satisfy.
    """
    if reduction_rate <= 1:
        raise ValueError("--de-reduction-rates values must be greater than 1")

    strata = pd.DataFrame(index=model.obs_names)
    for name in ["celltype", "donor", "condition"]:
        strata[name] = clean_obs_values(model.obs[keys[name]]).to_numpy()
    table = (
        strata.groupby(["celltype", "donor", "condition"], sort=True, dropna=False)
        .size()
        .rename("n_cells")
        .reset_index()
    )
    table.insert(0, "stratum_id", np.arange(len(table), dtype=int))
    counts = table["n_cells"].to_numpy(dtype=int)
    ideal = counts.astype(float) / float(reduction_rate)
    target = np.maximum(1, np.floor(ideal).astype(int))

    requested_total = int(np.floor(model.n_obs / float(reduction_rate) + 0.5))
    feasible_total = min(model.n_obs, max(len(table), requested_total))

    while int(target.sum()) < feasible_total:
        candidates = np.flatnonzero(target < counts)
        if len(candidates) == 0:
            break
        deficit = ideal[candidates] - target[candidates]
        chosen = candidates[int(np.argmax(deficit))]
        target[chosen] += 1

    while int(target.sum()) > feasible_total:
        candidates = np.flatnonzero(target > 1)
        if len(candidates) == 0:
            break
        excess = target[candidates] - ideal[candidates]
        chosen = candidates[int(np.argmax(excess))]
        target[chosen] -= 1

    if np.any(target < 1) or np.any(target > counts):
        raise RuntimeError("Internal error allocating feasible DE metacell targets")
    if int(target.sum()) != feasible_total:
        raise RuntimeError("Could not allocate the requested shared DE metacell budget")

    table["ideal_metacells_at_requested_rate"] = ideal
    table["target_metacells"] = target
    table["target_reduction_rate"] = counts / target
    table["requested_global_reduction_rate"] = float(reduction_rate)
    table["requested_global_metacells"] = requested_total
    table["realized_global_target_metacells"] = feasible_total
    return table


def exact_kmeans_labels(
    points: np.ndarray,
    n_clusters: int,
    seed: int,
    sample_weight: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Run deterministic K-means and guarantee exactly n_clusters labels."""
    points = np.asarray(points, dtype=np.float32)
    n_points = points.shape[0]
    if not 1 <= n_clusters <= n_points:
        raise ValueError(f"Cannot assign {n_points} points to {n_clusters} clusters")
    if n_clusters == 1:
        return np.zeros(n_points, dtype=int)
    if n_clusters == n_points:
        return np.arange(n_points, dtype=int)

    estimator = KMeans(
        n_clusters=n_clusters,
        init="k-means++",
        n_init=10,
        max_iter=300,
        algorithm="lloyd",
        random_state=seed,
    )
    if sample_weight is None:
        estimator.fit(points)
    else:
        estimator.fit(points, sample_weight=np.asarray(sample_weight, dtype=float))
    labels = np.asarray(estimator.labels_, dtype=int)

    # Coincident points can make sklearn return fewer than the requested number
    # of occupied clusters. Deterministically peel farthest points from clusters
    # containing more than one point so the comparison budget remains exact.
    present = set(np.unique(labels).tolist())
    missing = [cluster for cluster in range(n_clusters) if cluster not in present]
    for empty_cluster in missing:
        sizes = np.bincount(labels, minlength=n_clusters)
        candidates = np.flatnonzero(sizes[labels] > 1)
        if len(candidates) == 0:
            raise RuntimeError("Unable to repair an empty K-means cluster")
        centers = np.zeros((n_clusters, points.shape[1]), dtype=np.float64)
        for cluster in np.unique(labels):
            cluster_mask = labels == cluster
            if sample_weight is None:
                centers[cluster] = points[cluster_mask].mean(axis=0)
            else:
                centers[cluster] = np.average(
                    points[cluster_mask],
                    axis=0,
                    weights=np.asarray(sample_weight)[cluster_mask],
                )
        errors = np.square(points[candidates] - centers[labels[candidates]]).sum(axis=1)
        chosen = candidates[int(np.argmax(errors))]
        labels[chosen] = empty_cluster

    if np.unique(labels).size != n_clusters:
        raise RuntimeError("K-means repair did not realize the exact DE target")
    return labels


def allocate_fragment_split_targets(fragment_sizes: np.ndarray, total: int) -> np.ndarray:
    """Allocate an exact number of subclusters while preserving fragments."""
    fragment_sizes = np.asarray(fragment_sizes, dtype=int)
    if not len(fragment_sizes) <= total <= int(fragment_sizes.sum()):
        raise ValueError("Infeasible fragment split target")
    allocated = np.ones(len(fragment_sizes), dtype=int)
    ideal = fragment_sizes / fragment_sizes.sum() * total
    while int(allocated.sum()) < total:
        candidates = np.flatnonzero(allocated < fragment_sizes)
        deficit = ideal[candidates] - allocated[candidates]
        chosen = candidates[int(np.argmax(deficit))]
        allocated[chosen] += 1
    return allocated


def build_matched_de_assignment(
    model: ad.AnnData,
    keys: Mapping[str, str],
    global_assignment: pd.Series,
    method: str,
    target_table: pd.DataFrame,
    seed: int,
) -> Tuple[pd.Series, pd.DataFrame]:
    """Rebalance native partition fragments to a shared within-stratum target.

    Native global metacells are first cut only at cell-type x donor x condition
    boundaries. If a stratum contains too many native fragments, fragment PCA
    centroids are merged by size-weighted K-means. In the uncommon case that it
    contains too few, only existing fragments are split by within-fragment
    K-means. This identical post-processing preserves all cells and as much of
    each native partition as possible while making the effective compression
    exactly equal across methods.
    """
    aligned = global_assignment.reindex(model.obs_names)
    if aligned.isna().any():
        raise ValueError(f"{method}: global assignment cannot be aligned for DE")
    native_labels = aligned.astype(str).to_numpy()
    X_pca = np.asarray(model.obsm["X_pca"], dtype=np.float32)

    key_frame = pd.DataFrame(index=model.obs_names)
    for name in ["celltype", "donor", "condition"]:
        key_frame[name] = clean_obs_values(model.obs[keys[name]]).to_numpy()
    target_lookup = target_table.set_index(["celltype", "donor", "condition"])

    final_labels = np.empty(model.n_obs, dtype=object)
    final_labels[:] = None
    summary_rows = []
    grouped = key_frame.groupby(
        ["celltype", "donor", "condition"], sort=True, dropna=False
    ).indices
    for group_values, positions_raw in grouped.items():
        positions = np.asarray(positions_raw, dtype=int)
        target_row = target_lookup.loc[group_values]
        stratum_id = int(target_row["stratum_id"])
        target = int(target_row["target_metacells"])
        group_native = native_labels[positions]
        _, fragment_codes = np.unique(group_native, return_inverse=True)
        n_fragments = int(fragment_codes.max() + 1)
        fragment_sizes = np.bincount(fragment_codes, minlength=n_fragments)
        group_seed = stable_seed(
            seed, MATCHED_DE_PROTOCOL_VERSION, method, stratum_id
        )

        if n_fragments > target:
            centroids = np.zeros((n_fragments, X_pca.shape[1]), dtype=np.float64)
            np.add.at(centroids, fragment_codes, X_pca[positions])
            centroids /= fragment_sizes[:, None]
            fragment_to_group = exact_kmeans_labels(
                centroids,
                target,
                group_seed,
                sample_weight=fragment_sizes,
            )
            local_labels = fragment_to_group[fragment_codes]
            adjustment = "merge_native_fragments"
        elif n_fragments < target:
            split_targets = allocate_fragment_split_targets(fragment_sizes, target)
            local_labels = np.empty(len(positions), dtype=int)
            offset = 0
            for fragment in range(n_fragments):
                local_positions = np.flatnonzero(fragment_codes == fragment)
                n_subclusters = int(split_targets[fragment])
                split_labels = exact_kmeans_labels(
                    X_pca[positions[local_positions]],
                    n_subclusters,
                    stable_seed(group_seed, "split", fragment),
                )
                local_labels[local_positions] = split_labels + offset
                offset += n_subclusters
            adjustment = "split_native_fragments"
        else:
            local_labels = fragment_codes
            adjustment = "unchanged"

        realized = int(np.unique(local_labels).size)
        if realized != target:
            raise RuntimeError(
                f"{method} stratum {stratum_id}: realized {realized}, target {target}"
            )
        prefix = (
            f"{slugify(method).lower()}|{MATCHED_DE_PROTOCOL_VERSION}|"
            f"stratum={stratum_id}|mc="
        )
        final_labels[positions] = [f"{prefix}{label}" for label in local_labels]
        summary_rows.append(
            {
                "protocol": MATCHED_DE_PROTOCOL_VERSION,
                "method": method,
                "stratum_id": stratum_id,
                "celltype": group_values[0],
                "donor": group_values[1],
                "condition": group_values[2],
                "n_cells": len(positions),
                "native_global_fragments": n_fragments,
                "target_metacells": target,
                "realized_metacells": realized,
                "realized_reduction_rate": len(positions) / realized,
                "adjustment": adjustment,
                "native_fragments_merged": max(0, n_fragments - target),
                "native_fragments_split": max(0, target - n_fragments),
            }
        )

    if any(value is None for value in final_labels):
        raise RuntimeError(f"{method}: at least one cell lacks a matched DE assignment")
    assignment = pd.Series(final_labels.astype(str), index=model.obs_names, dtype="string")
    return assignment.astype(str), pd.DataFrame(summary_rows).sort_values("stratum_id")


def run_rank_genes_groups(
    adata: ad.AnnData,
    group_key: str,
    n_genes: int,
    groups: Optional[Sequence[str]] = None,
    reference: str = "rest",
) -> pd.DataFrame:
    work = adata.copy()
    work.obs[group_key] = pd.Categorical(clean_obs_values(work.obs[group_key]))
    requested_groups = "all" if groups is None else [str(value) for value in groups]
    sc.tl.rank_genes_groups(
        work,
        groupby=group_key,
        groups=requested_groups,
        reference=reference,
        method="wilcoxon",
        n_genes=min(n_genes, work.n_vars),
        use_raw=False,
        key_added="wilcoxon",
    )
    return sc.get.rank_genes_groups_df(work, key="wilcoxon", group=None)


def celltype_de(
    adata: ad.AnnData,
    celltype_key: str,
    condition_key: str,
    reference_condition: str,
    top_n: int,
) -> pd.DataFrame:
    keep = clean_obs_values(adata.obs[condition_key]) == reference_condition
    subset = adata[keep].copy()
    if subset.n_obs == 0:
        raise ValueError(f"No cells remain for reference condition {reference_condition}")
    if clean_obs_values(subset.obs[celltype_key]).nunique() < 2:
        raise ValueError("Cell-type DE requires at least two cell types")
    return run_rank_genes_groups(subset, celltype_key, top_n)


def condition_de(
    adata: ad.AnnData,
    celltype_key: str,
    condition_key: str,
    reference_condition: str,
) -> pd.DataFrame:
    frames = []
    celltypes = clean_obs_values(adata.obs[celltype_key])
    for celltype in celltypes.unique():
        subset = adata[celltypes == celltype].copy()
        subset_conditions = clean_obs_values(subset.obs[condition_key])
        if reference_condition not in subset_conditions.unique():
            logger.warning("Skipping %s: reference condition is absent", celltype)
            continue
        contrasts = [value for value in subset_conditions.unique() if value != reference_condition]
        for contrast in contrasts:
            comparison = subset[subset_conditions.isin([reference_condition, contrast])].copy()
            if clean_obs_values(comparison.obs[condition_key]).value_counts().min() < 2:
                logger.warning("Skipping %s/%s: fewer than two observations in one group", celltype, contrast)
                continue
            frame = run_rank_genes_groups(
                comparison,
                condition_key,
                comparison.n_vars,
                groups=[contrast],
                reference=reference_condition,
            )
            frame["celltype"] = celltype
            frame["contrast"] = contrast
            frame["reference"] = reference_condition
            frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def rank_concordance(
    original: pd.DataFrame,
    metacell: pd.DataFrame,
    method: str,
    top_n: int,
) -> pd.DataFrame:
    rows = []
    for group in sorted(set(original["group"]).intersection(metacell["group"])):
        left = original[original["group"] == group].head(top_n).reset_index(drop=True)
        right = metacell[metacell["group"] == group].head(top_n).reset_index(drop=True)
        left_rank = {gene: rank + 1 for rank, gene in enumerate(left["names"].astype(str))}
        right_rank = {gene: rank + 1 for rank, gene in enumerate(right["names"].astype(str))}
        genes = sorted(set(left_rank).union(right_rank))
        missing_rank = top_n + 1
        tau, p_value = kendalltau(
            [left_rank.get(gene, missing_rank) for gene in genes],
            [right_rank.get(gene, missing_rank) for gene in genes],
        )
        rows.append(
            {
                "method": method,
                "celltype": group,
                "kendall_tau": tau,
                "p_value": p_value,
                "top_n": top_n,
                "overlap": len(set(left_rank).intersection(right_rank)),
                "union": len(genes),
            }
        )
    return pd.DataFrame(rows)


def condition_correlations(
    original: pd.DataFrame,
    metacell: pd.DataFrame,
    method: str,
) -> pd.DataFrame:
    if original.empty or metacell.empty:
        return pd.DataFrame()
    keys = ["celltype", "contrast", "reference", "names"]
    merged = original.merge(
        metacell,
        on=keys,
        suffixes=("_original", "_metacell"),
    )
    rows = []
    for group_values, frame in merged.groupby(["celltype", "contrast", "reference"]):
        finite = np.isfinite(frame["logfoldchanges_original"]) & np.isfinite(
            frame["logfoldchanges_metacell"]
        )
        usable = frame.loc[finite]
        if len(usable) < 3:
            correlation, p_value = np.nan, np.nan
        else:
            correlation, p_value = pearsonr(
                usable["logfoldchanges_original"],
                usable["logfoldchanges_metacell"],
            )
        rows.append(
            {
                "method": method,
                "celltype": group_values[0],
                "contrast": group_values[1],
                "reference": group_values[2],
                "pearson_r": correlation,
                "p_value": p_value,
                "n_genes": len(usable),
            }
        )
    return pd.DataFrame(rows)


def remove_previous_de_outputs(out_dir: Path) -> None:
    """Remove only files generated by the old or current Fig. 5 stage."""
    patterns = [
        "*_stratified_assignment.csv.gz",
        "*_stratified_summary.csv",
        "*_stratified_metacells.h5ad",
        "*_matched10x_assignment.csv.gz",
        "*_matched10x_stratum_summary.csv",
        "*_matched10x_metacells.h5ad",
        "*_matched_assignment.csv.gz",
        "*_matched_stratum_summary.csv",
        "*_matched_metacells.h5ad",
        "*_celltype_de.csv.gz",
        "*_condition_de.csv.gz",
        "*_celltype_rank_concordance.csv",
        "*_condition_correlations.csv",
        "metacell_summary.csv",
        "shared_stratum_targets.csv",
        "all_methods_matched10x_stratum_summary.csv",
        "all_methods_matched_stratum_summary.csv",
        "de_fairness_validation.csv",
        "de_protocol.json",
    ]
    removed = 0
    for pattern in patterns:
        for path in out_dir.glob(pattern):
            if path.is_file():
                path.unlink()
                removed += 1
    logger.info("Removed %d previous Fig. 5 output files before replacement", removed)


def reduction_rate_slug(reduction_rate: float) -> str:
    """Return a stable path component such as requested_r10 or requested_r12p5."""
    value = f"{float(reduction_rate):.8g}".replace("-", "m").replace(".", "p")
    return f"requested_r{value}"


def run_fig5(
    args: argparse.Namespace,
    base: ad.AnnData,
    model: ad.AnnData,
    keys: Mapping[str, str],
    assignments: Mapping[str, pd.Series],
    reference_condition: str,
    out_dir: Path,
    reduction_rate: float,
    original_celltype_de: Optional[pd.DataFrame] = None,
    original_condition_de: Optional[pd.DataFrame] = None,
) -> None:
    safe_mkdir(out_dir)
    if args.force or args.force_de:
        remove_previous_de_outputs(out_dir)
    target_table = allocate_stratum_metacell_targets(
        model,
        keys,
        reduction_rate,
    )
    save_csv(target_table, out_dir / "shared_stratum_targets.csv")
    target_total = int(target_table["target_metacells"].sum())
    realized_global_rate = base.n_obs / target_total
    logger.info(
        "Fig. 5 fair protocol | requested reduction=%.3gx | cells=%d | "
        "strata=%d | shared target profiles=%d | realized reduction=%.4gx",
        reduction_rate,
        base.n_obs,
        len(target_table),
        target_total,
        realized_global_rate,
    )

    protocol_metadata = {
        "protocol": MATCHED_DE_PROTOCOL_VERSION,
        "description": (
            "Every method uses the same evaluation cells and the same exact "
            "celltype-by-donor-by-condition metacell target. Released native "
            "global partition fragments are rebalanced within each stratum only."
        ),
        "requested_reduction_rate": reduction_rate,
        "source_global_requested_metacells": args.fig4_primary_target,
        "evaluation_cells": base.n_obs,
        "n_strata": len(target_table),
        "shared_target_metacells": target_total,
        "realized_global_reduction_rate": realized_global_rate,
        "reference_condition": reference_condition,
        "random_seed": args.random_seed,
        "posthoc_kmeans_used": True,
        "comparison_axis": "shared_target_metacells",
    }
    save_json(protocol_metadata, out_dir / "de_protocol.json")

    original = base.copy()
    original.obs["celltype"] = clean_obs_values(base.obs[keys["celltype"]]).to_numpy()
    original.obs["donor"] = clean_obs_values(base.obs[keys["donor"]]).to_numpy()
    original.obs["condition"] = clean_obs_values(base.obs[keys["condition"]]).to_numpy()

    if original_celltype_de is None:
        logger.info("Fig. 5 | original-cell cell-type DE")
        original_celltype_de = celltype_de(
            original,
            "celltype",
            "condition",
            reference_condition,
            args.de_top_genes,
        )
        original_celltype_de["method"] = "Original cells"
    save_csv(original_celltype_de, out_dir / "original_celltype_de.csv.gz")

    if original_condition_de is None:
        logger.info("Fig. 5 | original-cell condition DE")
        original_condition_de = condition_de(
            original,
            "celltype",
            "condition",
            reference_condition,
        )
        original_condition_de["method"] = "Original cells"
    save_csv(original_condition_de, out_dir / "original_condition_de.csv.gz")

    summary_rows = []
    stratum_summaries = []
    no_cross_stratum_by_method: Dict[str, bool] = {}
    for method, global_assignment in assignments.items():
        method_slug = slugify(method).lower()
        logger.info(
            "Fig. 5 | matching %s to the shared within-stratum %.3gx target",
            method,
            reduction_rate,
        )
        assignment_path = out_dir / f"{method_slug}_matched_assignment.csv.gz"
        strata_path = out_dir / f"{method_slug}_matched_stratum_summary.csv"
        if (
            assignment_path.exists()
            and strata_path.exists()
            and not (args.force or args.force_de)
        ):
            assignment_frame = pd.read_csv(assignment_path)
            assignment = pd.Series(
                assignment_frame["metacell"].astype(str).to_numpy(),
                index=assignment_frame["cell_id"].astype(str),
            ).reindex(base.obs_names)
            if assignment.isna().any():
                raise ValueError(
                    f"Cached DE assignment cannot be aligned: {assignment_path}. "
                    "Delete it or rerun with --force-de."
                )
            strata_summary = pd.read_csv(strata_path)
            cache_is_current = (
                "protocol" in strata_summary.columns
                and set(strata_summary["protocol"].astype(str))
                == {MATCHED_DE_PROTOCOL_VERSION}
                and int(strata_summary["realized_metacells"].sum()) == target_total
                and np.array_equal(
                    strata_summary.sort_values("stratum_id")["target_metacells"].to_numpy(),
                    target_table.sort_values("stratum_id")["target_metacells"].to_numpy(),
                )
            )
            if not cache_is_current:
                raise ValueError(
                    f"Stale or incompatible fair-DE cache: {assignment_path}. "
                    "Rerun with --force-de."
                )
            logger.info("Using cached matched-compression assignment: %s", assignment_path)
        else:
            assignment, strata_summary = build_matched_de_assignment(
                model,
                keys,
                global_assignment,
                method,
                target_table,
                args.random_seed,
            )
            save_csv(
                pd.DataFrame({"cell_id": assignment.index, "metacell": assignment.values}),
                assignment_path,
            )
            save_csv(strata_summary, strata_path)
        stratum_summaries.append(strata_summary)

        assignment_audit = original.obs[["celltype", "donor", "condition"]].copy()
        assignment_audit["metacell"] = assignment.reindex(original.obs_names).to_numpy()
        key_counts = assignment_audit.groupby("metacell", sort=False)[
            ["celltype", "donor", "condition"]
        ].nunique()
        no_cross_strata = bool((key_counts <= 1).all().all())
        no_cross_stratum_by_method[method] = no_cross_strata
        if not no_cross_strata:
            raise RuntimeError(f"{method}: a matched DE metacell crosses a DE stratum")

        meta, _ = aggregate_metacells(
            original,
            assignment,
            {"celltype": "celltype", "donor": "donor", "condition": "condition"},
            method,
        )
        if meta.n_obs != target_total:
            raise RuntimeError(
                f"{method}: aggregated {meta.n_obs} metacells; shared target is {target_total}"
            )
        meta_path = out_dir / f"{method_slug}_matched_metacells.h5ad"
        meta.write_h5ad(meta_path, compression="gzip")

        summary_rows.append(
            {
                "method": method,
                "n_cells": original.n_obs,
                "n_metacells": meta.n_obs,
                "effective_reduction_rate": original.n_obs / meta.n_obs,
                "source_gamma": args.fig4_gamma,
                "requested_de_reduction_rate": reduction_rate,
                "shared_target_metacells": target_total,
                "source_global_requested_metacells": args.fig4_primary_target,
                "source_stratified_fragments_before_matching": int(
                    strata_summary["native_global_fragments"].sum()
                ),
                "n_merge_strata": int(
                    (strata_summary["adjustment"] == "merge_native_fragments").sum()
                ),
                "n_split_strata": int(
                    (strata_summary["adjustment"] == "split_native_fragments").sum()
                ),
                "n_unchanged_strata": int(
                    (strata_summary["adjustment"] == "unchanged").sum()
                ),
                "de_protocol": MATCHED_DE_PROTOCOL_VERSION,
                "n_strata": len(strata_summary),
                "all_cells_covered": int(assignment.notna().sum()) == original.n_obs,
                "all_stratum_targets_matched": bool(
                    (strata_summary["target_metacells"] == strata_summary["realized_metacells"]).all()
                ),
                "no_cross_stratum_metacells": no_cross_strata,
            }
        )

        method_celltype_de = celltype_de(
            meta,
            "celltype",
            "condition",
            reference_condition,
            args.de_top_genes,
        )
        method_celltype_de["method"] = method
        method_celltype_de["requested_de_reduction_rate"] = reduction_rate
        method_celltype_de["shared_target_metacells"] = target_total
        method_celltype_de["effective_reduction_rate"] = realized_global_rate
        save_csv(method_celltype_de, out_dir / f"{method_slug}_celltype_de.csv.gz")
        concordance = rank_concordance(
            original_celltype_de,
            method_celltype_de,
            method,
            args.de_top_genes,
        )
        concordance["requested_de_reduction_rate"] = reduction_rate
        concordance["shared_target_metacells"] = target_total
        concordance["effective_reduction_rate"] = realized_global_rate
        save_csv(concordance, out_dir / f"{method_slug}_celltype_rank_concordance.csv")

        method_condition_de = condition_de(
            meta,
            "celltype",
            "condition",
            reference_condition,
        )
        method_condition_de["method"] = method
        method_condition_de["requested_de_reduction_rate"] = reduction_rate
        method_condition_de["shared_target_metacells"] = target_total
        method_condition_de["effective_reduction_rate"] = realized_global_rate
        save_csv(method_condition_de, out_dir / f"{method_slug}_condition_de.csv.gz")
        correlations = condition_correlations(
            original_condition_de,
            method_condition_de,
            method,
        )
        correlations["requested_de_reduction_rate"] = reduction_rate
        correlations["shared_target_metacells"] = target_total
        correlations["effective_reduction_rate"] = realized_global_rate
        save_csv(correlations, out_dir / f"{method_slug}_condition_correlations.csv")
        logger.info("Fig. 5 | finished %s", method)

    summary = pd.DataFrame(summary_rows)
    all_strata = pd.concat(stratum_summaries, ignore_index=True)
    validation_rows = []
    expected_targets = target_table.set_index("stratum_id")["target_metacells"].sort_index()
    for method, frame in all_strata.groupby("method", sort=False):
        realized = frame.set_index("stratum_id")["realized_metacells"].sort_index()
        aligned_realized = realized.reindex(expected_targets.index)
        per_stratum_match = bool(
            aligned_realized.notna().all()
            and np.array_equal(
                aligned_realized.to_numpy(dtype=int),
                expected_targets.to_numpy(dtype=int),
            )
        )
        validation_rows.append(
            {
                "method": method,
                "protocol": MATCHED_DE_PROTOCOL_VERSION,
                "requested_de_reduction_rate": reduction_rate,
                "evaluation_cells": original.n_obs,
                "expected_metacells": target_total,
                "realized_metacells": int(frame["realized_metacells"].sum()),
                "source_stratified_fragments_before_matching": int(
                    frame["native_global_fragments"].sum()
                ),
                "all_strata_present": len(frame) == len(target_table),
                "every_stratum_matches_shared_target": per_stratum_match,
                "no_cross_stratum_metacells": no_cross_stratum_by_method[method],
                "passed": (
                    len(frame) == len(target_table)
                    and int(frame["realized_metacells"].sum()) == target_total
                    and per_stratum_match
                    and no_cross_stratum_by_method[method]
                ),
            }
        )
    validation = pd.DataFrame(validation_rows)
    save_csv(summary, out_dir / "metacell_summary.csv")
    save_csv(all_strata, out_dir / "all_methods_matched_stratum_summary.csv")
    save_csv(validation, out_dir / "de_fairness_validation.csv")
    if not validation["passed"].all():
        failures = validation.loc[~validation["passed"], "method"].tolist()
        raise RuntimeError(f"Fair-DE validation failed for: {failures}")
    logger.info(
        "Fig. 5 fairness validation passed for all %d methods at %d profiles each",
        len(validation),
        target_total,
    )


def de_matched_rate_is_complete(
    out_dir: Path,
    methods: Sequence[str],
    reduction_rate: float,
    reference_condition: str,
) -> bool:
    """Check that one matched-compression DE rate has all validated outputs."""
    protocol_path = out_dir / "de_protocol.json"
    validation_path = out_dir / "de_fairness_validation.csv"
    if not protocol_path.is_file() or not validation_path.is_file():
        return False
    try:
        protocol = json.loads(protocol_path.read_text())
        validation = pd.read_csv(validation_path)
    except (OSError, json.JSONDecodeError, pd.errors.ParserError):
        return False
    if (
        protocol.get("protocol") != MATCHED_DE_PROTOCOL_VERSION
        or protocol.get("posthoc_kmeans_used") is not True
        or not np.isclose(
            float(protocol.get("requested_reduction_rate", np.nan)),
            float(reduction_rate),
        )
        or str(protocol.get("reference_condition")) != str(reference_condition)
        or set(validation.get("method", pd.Series(dtype=str)).astype(str))
        != set(methods)
        or not bool(validation.get("passed", pd.Series(dtype=bool)).all())
    ):
        return False
    required = [
        protocol_path,
        validation_path,
        out_dir / "shared_stratum_targets.csv",
        out_dir / "metacell_summary.csv",
        out_dir / "all_methods_matched_stratum_summary.csv",
        out_dir / "original_celltype_de.csv.gz",
        out_dir / "original_condition_de.csv.gz",
    ]
    for method in methods:
        slug = slugify(method).lower()
        required.extend(
            [
                out_dir / f"{slug}_matched_assignment.csv.gz",
                out_dir / f"{slug}_matched_stratum_summary.csv",
                out_dir / f"{slug}_matched_metacells.h5ad",
                out_dir / f"{slug}_celltype_de.csv.gz",
                out_dir / f"{slug}_condition_de.csv.gz",
                out_dir / f"{slug}_celltype_rank_concordance.csv",
                out_dir / f"{slug}_condition_correlations.csv",
            ]
        )
    return all(path.is_file() for path in required)


def run_fig5_matched_grid(
    args: argparse.Namespace,
    base: ad.AnnData,
    model: ad.AnnData,
    keys: Mapping[str, str],
    assignments: Mapping[str, pd.Series],
    reference_condition: str,
    out_root: Path,
) -> None:
    """Evaluate every method at identical within-stratum compression rates."""
    safe_mkdir(out_root)
    reference_dir = out_root / "original_cell_reference"
    safe_mkdir(reference_dir)
    original_celltype_path = reference_dir / "original_celltype_de.csv.gz"
    original_condition_path = reference_dir / "original_condition_de.csv.gz"

    original = base.copy()
    original.obs["celltype"] = clean_obs_values(base.obs[keys["celltype"]]).to_numpy()
    original.obs["donor"] = clean_obs_values(base.obs[keys["donor"]]).to_numpy()
    original.obs["condition"] = clean_obs_values(base.obs[keys["condition"]]).to_numpy()
    if (
        original_celltype_path.is_file()
        and original_condition_path.is_file()
        and not (args.force or args.force_de)
    ):
        original_celltype_de = pd.read_csv(original_celltype_path)
        original_condition_de = pd.read_csv(original_condition_path)
    else:
        logger.info("Matched Fig. 5 grid | computing one shared original-cell DE reference")
        original_celltype_de = celltype_de(
            original,
            "celltype",
            "condition",
            reference_condition,
            args.de_top_genes,
        )
        original_celltype_de["method"] = "Original cells"
        original_condition_de = condition_de(
            original,
            "celltype",
            "condition",
            reference_condition,
        )
        original_condition_de["method"] = "Original cells"
        save_csv(original_celltype_de, original_celltype_path)
        save_csv(original_condition_de, original_condition_path)

    for reduction_rate in args.de_reduction_rates:
        target_dir = out_root / reduction_rate_slug(reduction_rate)
        if not (args.force or args.force_de) and de_matched_rate_is_complete(
            target_dir,
            list(assignments),
            reduction_rate,
            reference_condition,
        ):
            logger.info(
                "Matched Fig. 5 %.3gx is complete; keeping validated checkpoints",
                reduction_rate,
            )
            continue
        run_fig5(
            args=args,
            base=base,
            model=model,
            keys=keys,
            assignments=assignments,
            reference_condition=reference_condition,
            out_dir=target_dir,
            reduction_rate=reduction_rate,
            original_celltype_de=original_celltype_de,
            original_condition_de=original_condition_de,
        )


def intersect_native_partition_with_de_strata(
    model: ad.AnnData,
    keys: Mapping[str, str],
    global_assignment: pd.Series,
    method: str,
    requested_metacells: int,
) -> Tuple[pd.Series, pd.DataFrame]:
    """Cut native metacells at DE boundaries without clustering or rebalancing."""
    aligned = global_assignment.reindex(model.obs_names)
    if aligned.isna().any():
        raise ValueError(f"{method}: native assignment cannot be aligned for DE")

    strata = pd.DataFrame(index=model.obs_names)
    for name in ["celltype", "donor", "condition"]:
        strata[name] = clean_obs_values(model.obs[keys[name]]).to_numpy()
    stratum_index = pd.MultiIndex.from_frame(strata)
    stratum_codes, stratum_levels = pd.factorize(stratum_index, sort=True)
    native = aligned.astype(str).to_numpy()
    final = pd.Series(
        [
            f"{slugify(method).lower()}|native={native_value}|stratum={stratum_code}"
            for native_value, stratum_code in zip(native, stratum_codes)
        ],
        index=model.obs_names,
        dtype="string",
    ).astype(str)

    rows: List[Dict[str, object]] = []
    for stratum_code, level in enumerate(stratum_levels):
        positions = np.flatnonzero(stratum_codes == stratum_code)
        native_fragments = int(np.unique(native[positions]).size)
        rows.append(
            {
                "protocol": NATIVE_DE_PROTOCOL_VERSION,
                "method": method,
                "requested_global_metacells": requested_metacells,
                "source_global_metacells": int(aligned.nunique()),
                "stratum_id": stratum_code,
                "celltype": str(level[0]),
                "donor": str(level[1]),
                "condition": str(level[2]),
                "n_cells": len(positions),
                "native_fragments": native_fragments,
                "realized_profiles": native_fragments,
                "realized_reduction_rate": len(positions) / native_fragments,
                "operation": "native_partition_intersection_only",
                "native_fragments_merged": 0,
                "native_fragments_reclustered": 0,
                "posthoc_kmeans_used": False,
            }
        )
    return final, pd.DataFrame(rows).sort_values("stratum_id")


def de_native_target_is_complete(
    out_dir: Path,
    methods: Sequence[str],
    requested_metacells: Optional[int] = None,
    reference_condition: Optional[str] = None,
) -> bool:
    protocol_path = out_dir / "de_protocol.json"
    if not protocol_path.is_file():
        return False
    try:
        protocol = json.loads(protocol_path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    if (
        protocol.get("protocol") != NATIVE_DE_PROTOCOL_VERSION
        or protocol.get("posthoc_kmeans_used") is not False
        or (
            requested_metacells is not None
            and int(protocol.get("requested_global_metacells", -1))
            != requested_metacells
        )
        or (
            reference_condition is not None
            and str(protocol.get("reference_condition")) != reference_condition
        )
    ):
        return False
    required = [protocol_path, out_dir / "metacell_summary.csv"]
    for method in methods:
        slug = slugify(method).lower()
        required.extend(
            [
                out_dir / f"{slug}_native_stratified_assignment.csv.gz",
                out_dir / f"{slug}_native_stratum_summary.csv",
                out_dir / f"{slug}_native_stratified_metacells.h5ad",
                out_dir / f"{slug}_celltype_de.csv.gz",
                out_dir / f"{slug}_condition_de.csv.gz",
                out_dir / f"{slug}_celltype_rank_concordance.csv",
                out_dir / f"{slug}_condition_correlations.csv",
            ]
        )
    return bool(required) and all(path.is_file() for path in required)


def run_fig5_native_grid(
    args: argparse.Namespace,
    base: ad.AnnData,
    model: ad.AnnData,
    keys: Mapping[str, str],
    assignments_by_target: Mapping[int, Mapping[str, pd.Series]],
    reference_condition: str,
    out_root: Path,
) -> None:
    """Evaluate DE preservation for the same native partition grid as Fig. 4."""
    safe_mkdir(out_root)
    original = base.copy()
    original.obs["celltype"] = clean_obs_values(base.obs[keys["celltype"]]).to_numpy()
    original.obs["donor"] = clean_obs_values(base.obs[keys["donor"]]).to_numpy()
    original.obs["condition"] = clean_obs_values(base.obs[keys["condition"]]).to_numpy()

    reference_dir = out_root / "original_cell_reference"
    safe_mkdir(reference_dir)
    original_celltype_path = reference_dir / "original_celltype_de.csv.gz"
    original_condition_path = reference_dir / "original_condition_de.csv.gz"
    if original_celltype_path.is_file() and original_condition_path.is_file() and not (
        args.force or args.force_de
    ):
        original_celltype_de = pd.read_csv(original_celltype_path)
        original_condition_de = pd.read_csv(original_condition_path)
    else:
        logger.info("Fig. 5 native grid | computing original-cell DE reference")
        original_celltype_de = celltype_de(
            original,
            "celltype",
            "condition",
            reference_condition,
            args.de_top_genes,
        )
        original_celltype_de["method"] = "Original cells"
        save_csv(original_celltype_de, original_celltype_path)
        original_condition_de = condition_de(
            original,
            "celltype",
            "condition",
            reference_condition,
        )
        original_condition_de["method"] = "Original cells"
        save_csv(original_condition_de, original_condition_path)

    for requested in sorted(int(value) for value in assignments_by_target):
        assignments = assignments_by_target[requested]
        target_dir = out_root / f"requested_m{requested}"
        safe_mkdir(target_dir)
        if not (args.force or args.force_de) and de_native_target_is_complete(
            target_dir,
            list(assignments),
            requested_metacells=requested,
            reference_condition=reference_condition,
        ):
            logger.info(
                "Fig. 5 requested m=%d is complete; keeping native-grid checkpoints",
                requested,
            )
            continue

        save_csv(original_celltype_de, target_dir / "original_celltype_de.csv.gz")
        save_csv(original_condition_de, target_dir / "original_condition_de.csv.gz")
        summary_rows: List[Dict[str, object]] = []
        all_strata: List[pd.DataFrame] = []
        validation_rows: List[Dict[str, object]] = []

        for method, global_assignment in assignments.items():
            slug = slugify(method).lower()
            source_count = int(global_assignment.reindex(model.obs_names).nunique())
            logger.info(
                "Fig. 5 native grid | %s requested=%d source=%d",
                method,
                requested,
                source_count,
            )
            assignment, strata_summary = intersect_native_partition_with_de_strata(
                model,
                keys,
                global_assignment,
                method,
                requested,
            )
            assignment_path = target_dir / f"{slug}_native_stratified_assignment.csv.gz"
            strata_path = target_dir / f"{slug}_native_stratum_summary.csv"
            save_csv(
                pd.DataFrame(
                    {"cell_id": assignment.index.astype(str), "metacell": assignment.to_numpy()}
                ),
                assignment_path,
            )
            save_csv(strata_summary, strata_path)
            all_strata.append(strata_summary)

            audit = original.obs[["celltype", "donor", "condition"]].copy()
            audit["metacell"] = assignment.reindex(original.obs_names).to_numpy()
            key_counts = audit.groupby("metacell", sort=False)[
                ["celltype", "donor", "condition"]
            ].nunique()
            no_cross_strata = bool((key_counts <= 1).all().all())
            if not no_cross_strata:
                raise RuntimeError(f"{method}: a native DE profile crosses a DE stratum")

            meta, _ = aggregate_metacells(
                original,
                assignment,
                {"celltype": "celltype", "donor": "donor", "condition": "condition"},
                method,
            )
            realized_profiles = int(meta.n_obs)
            meta.write_h5ad(
                target_dir / f"{slug}_native_stratified_metacells.h5ad",
                compression="gzip",
            )

            shared_annotations = {
                "requested_global_metacells": requested,
                "source_global_metacells": source_count,
                "realized_stratified_profiles": realized_profiles,
                "effective_reduction_rate": original.n_obs / realized_profiles,
                "de_protocol": NATIVE_DE_PROTOCOL_VERSION,
            }
            method_celltype_de = celltype_de(
                meta,
                "celltype",
                "condition",
                reference_condition,
                args.de_top_genes,
            )
            method_celltype_de["method"] = method
            for key, value in shared_annotations.items():
                method_celltype_de[key] = value
            save_csv(method_celltype_de, target_dir / f"{slug}_celltype_de.csv.gz")
            concordance = rank_concordance(
                original_celltype_de,
                method_celltype_de,
                method,
                args.de_top_genes,
            )
            for key, value in shared_annotations.items():
                concordance[key] = value
            save_csv(concordance, target_dir / f"{slug}_celltype_rank_concordance.csv")

            method_condition_de = condition_de(
                meta,
                "celltype",
                "condition",
                reference_condition,
            )
            method_condition_de["method"] = method
            for key, value in shared_annotations.items():
                method_condition_de[key] = value
            save_csv(method_condition_de, target_dir / f"{slug}_condition_de.csv.gz")
            correlations = condition_correlations(
                original_condition_de,
                method_condition_de,
                method,
            )
            for key, value in shared_annotations.items():
                correlations[key] = value
            save_csv(correlations, target_dir / f"{slug}_condition_correlations.csv")

            summary_rows.append(
                {
                    "protocol": NATIVE_DE_PROTOCOL_VERSION,
                    "method": method,
                    "n_cells": original.n_obs,
                    "requested_global_metacells": requested,
                    "source_global_metacells": source_count,
                    "realized_stratified_profiles": realized_profiles,
                    "effective_reduction_rate": original.n_obs / realized_profiles,
                    "n_strata": len(strata_summary),
                    "all_cells_covered": int(assignment.notna().sum()) == original.n_obs,
                    "no_cross_stratum_metacells": no_cross_strata,
                    "posthoc_kmeans_used": False,
                }
            )
            passed = (
                int(assignment.notna().sum()) == original.n_obs
                and no_cross_strata
                and int(strata_summary["realized_profiles"].sum()) == realized_profiles
                and not bool(strata_summary["posthoc_kmeans_used"].any())
            )
            validation_rows.append(
                {
                    "protocol": NATIVE_DE_PROTOCOL_VERSION,
                    "method": method,
                    "requested_global_metacells": requested,
                    "source_global_metacells": source_count,
                    "realized_stratified_profiles": realized_profiles,
                    "all_cells_covered": int(assignment.notna().sum()) == original.n_obs,
                    "no_cross_stratum_metacells": no_cross_strata,
                    "posthoc_kmeans_used": False,
                    "passed": passed,
                }
            )

        summary = pd.DataFrame(summary_rows)
        validation = pd.DataFrame(validation_rows)
        save_csv(summary, target_dir / "metacell_summary.csv")
        save_csv(pd.concat(all_strata, ignore_index=True), target_dir / "all_methods_native_stratum_summary.csv")
        save_csv(validation, target_dir / "de_protocol_validation.csv")
        save_json(
            {
                "protocol": NATIVE_DE_PROTOCOL_VERSION,
                "description": (
                    "The same native global partitions used for Fig. 4 are intersected "
                    "with celltype-by-donor-by-condition boundaries. No method is merged, "
                    "split by K-means, reclustered, or forced to an exact post-stratification count."
                ),
                "requested_global_metacells": requested,
                "evaluation_cells": original.n_obs,
                "reference_condition": reference_condition,
                "posthoc_kmeans_used": False,
                "comparison_axis": "realized_stratified_profiles",
            },
            target_dir / "de_protocol.json",
        )
        if not bool(validation["passed"].all()):
            failures = validation.loc[~validation["passed"], "method"].tolist()
            raise RuntimeError(f"Native DE protocol validation failed for: {failures}")


# =========================================================
# Plot-only layer (reads CSV checkpoints)
# =========================================================
def plot_umap_grid(
    frame: pd.DataFrame,
    output_path: Path,
    title_prefix: str,
    point_size: float,
    dpi: int,
) -> None:
    if frame.empty:
        logger.warning("No UMAP CSVs found for %s", output_path)
        return
    methods = frame["method"].drop_duplicates().tolist()
    fig, axes = plt.subplots(len(methods), 2, figsize=(12, 5 * len(methods)), squeeze=False)
    for row, method in enumerate(methods):
        subset = frame[frame["method"] == method]
        for column, color_key in enumerate(["celltype", "batch"]):
            ax = axes[row, column]
            sns.scatterplot(
                data=subset,
                x="UMAP1",
                y="UMAP2",
                hue=color_key,
                s=point_size,
                linewidth=0,
                alpha=0.8,
                rasterized=True,
                ax=ax,
            )
            ax.set_title(f"{title_prefix}: {method} — {color_key}")
            ax.set_xticks([])
            ax.set_yticks([])
            if ax.legend_ is not None:
                ax.legend_.remove()
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved figure: %s", output_path)


def plot_umap_individuals(
    frame: pd.DataFrame,
    output_dir: Path,
    title_prefix: str,
    point_size: float,
    dpi: int,
    color_keys: Sequence[str],
) -> None:
    """Save every method/color combination as its own publication PNG."""
    if frame.empty:
        logger.warning("No UMAP CSVs found for individual plots in %s", output_dir)
        return
    safe_mkdir(output_dir)
    for method in frame["method"].drop_duplicates():
        subset = frame[frame["method"] == method]
        for color_key in color_keys:
            if color_key not in subset.columns:
                continue
            fig, ax = plt.subplots(figsize=(6.5, 6.0))
            sns.scatterplot(
                data=subset,
                x="UMAP1",
                y="UMAP2",
                hue=color_key,
                s=point_size,
                linewidth=0,
                alpha=0.85,
                rasterized=True,
                ax=ax,
            )
            ax.set_title(f"{title_prefix}: {method} — {color_key}")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel("UMAP 1")
            ax.set_ylabel("UMAP 2")
            if ax.legend_ is not None:
                ax.legend(
                    title=color_key,
                    bbox_to_anchor=(1.02, 1.0),
                    loc="upper left",
                    borderaxespad=0,
                    frameon=False,
                    markerscale=2,
                )
            fig.tight_layout()
            output_path = output_dir / (
                f"{slugify(method).lower()}__colored_by_{slugify(color_key).lower()}.png"
            )
            fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
            plt.close(fig)
            logger.info("Saved individual figure: %s", output_path)


def plot_fig4(intermediate_dir: Path, fig4_dir: Path, dpi: int) -> None:
    individual_dir = fig4_dir / "individual_panels"
    overview_dir = fig4_dir / "overview_panels"
    for path in [fig4_dir, individual_dir, overview_dir]:
        safe_mkdir(path)
    unintegrated = read_csvs(intermediate_dir.glob("*_unintegrated_umap.csv.gz"))
    recovered = read_csvs(intermediate_dir.glob("*_recovered_umap.csv.gz"))
    metacells = read_csvs(intermediate_dir.glob("*_metacell_umap.csv.gz"))
    plot_umap_individuals(
        unintegrated,
        individual_dir / "umap_original_before_harmony",
        "Original cells before Harmony",
        2.0,
        dpi,
        ["celltype", "batch"],
    )
    plot_umap_individuals(
        recovered,
        individual_dir / "umap_integrated_or_recovered_cells",
        "Integrated or recovered cells",
        2.0,
        dpi,
        ["celltype", "batch"],
    )
    plot_umap_individuals(
        metacells,
        individual_dir / "umap_integrated_metacells",
        "Integrated metacells",
        10.0,
        dpi,
        ["celltype", "batch"],
    )
    plot_umap_grid(
        unintegrated,
        overview_dir / "fig4_original_unintegrated_umap_overview.png",
        "Original cells before Harmony",
        2.0,
        dpi,
    )
    plot_umap_grid(
        recovered,
        overview_dir / "fig4_recovered_cells_umap_overview.png",
        "Recovered cells",
        2.0,
        dpi,
    )
    plot_umap_grid(
        metacells,
        overview_dir / "fig4_integrated_metacells_umap_overview.png",
        "Integrated metacells",
        10.0,
        dpi,
    )

    metrics = read_csvs(intermediate_dir.glob("*_clustering_metrics.csv"))
    if not metrics.empty:
        grid = sns.relplot(
            data=metrics,
            x="resolution",
            y="score",
            hue="method",
            style="representation",
            col="metric",
            kind="line",
            marker="o",
            facet_kws={"sharex": False},
            palette=CUSTOM_PALETTE,
            height=4,
            aspect=1.05,
        )
        grid.set(ylim=(0, 1))
        grid.fig.savefig(
            overview_dir / "fig4_clustering_metrics_overview.png",
            dpi=dpi,
            bbox_inches="tight",
        )
        plt.close(grid.fig)
        metric_dir = individual_dir / "clustering_metrics"
        safe_mkdir(metric_dir)
        for representation in metrics["representation"].drop_duplicates():
            for metric in metrics["metric"].drop_duplicates():
                subset = metrics[
                    (metrics["representation"] == representation)
                    & (metrics["metric"] == metric)
                ]
                if subset.empty:
                    continue
                fig, ax = plt.subplots(figsize=(6.5, 5.0))
                sns.lineplot(
                    data=subset,
                    x="resolution",
                    y="score",
                    hue="method",
                    marker="o",
                    palette=CUSTOM_PALETTE,
                    ax=ax,
                )
                ax.set_ylim(0, 1.02)
                ax.set_title(f"{metric} — {representation}")
                ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
                fig.tight_layout()
                output_path = metric_dir / (
                    f"{slugify(representation).lower()}__{slugify(metric).lower()}.png"
                )
                fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
                plt.close(fig)

    lisi = read_csvs(intermediate_dir.glob("*_lisi_summary.csv"))
    if not lisi.empty:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax, metric in zip(
            axes, ["one_minus_cLISI_normalized", "iLISI_normalized"]
        ):
            subset = lisi[lisi["metric"] == metric]
            sns.boxplot(
                data=subset,
                x="method",
                y="score",
                hue="method",
                palette=CUSTOM_PALETTE,
                legend=False,
                ax=ax,
            )
            ax.set_title(metric.replace("_normalized", ""))
            ax.tick_params(axis="x", rotation=30)
            ax.set_ylim(0, 1.05)
        fig.tight_layout()
        fig.savefig(
            overview_dir / "fig4_lisi_metrics_overview.png",
            dpi=dpi,
            bbox_inches="tight",
            facecolor="white",
        )
        plt.close(fig)
        lisi_dir = individual_dir / "lisi_metrics"
        safe_mkdir(lisi_dir)
        for metric in ["one_minus_cLISI_normalized", "iLISI_normalized"]:
            subset = lisi[lisi["metric"] == metric]
            if subset.empty:
                continue
            fig, ax = plt.subplots(figsize=(6.5, 5.0))
            sns.boxplot(
                data=subset,
                x="method",
                y="score",
                hue="method",
                palette=CUSTOM_PALETTE,
                legend=False,
                ax=ax,
            )
            sns.stripplot(
                data=subset,
                x="method",
                y="score",
                color="black",
                size=2.5,
                alpha=0.5,
                ax=ax,
            )
            ax.set_title(metric.replace("_normalized", ""))
            ax.tick_params(axis="x", rotation=30)
            ax.set_ylim(0, 1.05)
            fig.tight_layout()
            output_path = lisi_dir / f"{slugify(metric).lower()}.png"
            fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
            plt.close(fig)


def plot_matched_fig4(
    intermediate_root: Path,
    figure_root: Path,
    targets: Sequence[int],
    primary_target: int,
    dpi: int,
    dataset_display_name: str = "PBMC",
) -> None:
    """Render every exact-count run plus cross-count sensitivity panels."""
    safe_mkdir(figure_root)
    targets = [int(target) for target in targets]
    for target in targets:
        target_intermediate = intermediate_root / f"m{target}"
        if not target_intermediate.is_dir():
            logger.warning("No Fig. 4 checkpoint directory for m=%d", target)
            continue
        plot_fig4(target_intermediate, figure_root / f"m{target}", dpi)

    all_metrics = []
    all_lisi = []
    for target in targets:
        target_dir = intermediate_root / f"m{target}"
        metrics = read_csvs(target_dir.glob("*_clustering_metrics.csv"))
        if not metrics.empty:
            metrics["target_metacells"] = target
            all_metrics.append(metrics)
        lisi = read_csvs(target_dir.glob("*_lisi_summary.csv"))
        if not lisi.empty:
            lisi["target_metacells"] = target
            all_lisi.append(lisi)

    sensitivity_dir = figure_root / "matched_count_sensitivity"
    safe_mkdir(sensitivity_dir)
    if all_metrics:
        metrics = pd.concat(all_metrics, ignore_index=True)
        save_csv(metrics, intermediate_root / "fig4_matched_count_clustering_metrics_all.csv")
        method_metrics = metrics[metrics["method"] != "Original cells"].copy()
        metric_means = (
            method_metrics.groupby(
                ["target_metacells", "method", "representation", "metric"],
                as_index=False,
            )["score"]
            .mean()
            .rename(columns={"score": "mean_score_across_clustering_resolutions"})
        )
        save_csv(
            metric_means,
            intermediate_root / "fig4_matched_count_clustering_metric_means.csv",
        )
        if not metric_means.empty:
            grid = sns.relplot(
                data=metric_means,
                x="target_metacells",
                y="mean_score_across_clustering_resolutions",
                hue="method",
                row="representation",
                col="metric",
                kind="line",
                marker="o",
                palette=CUSTOM_PALETTE,
                height=3.6,
                aspect=1.05,
                facet_kws={"sharey": True},
            )
            grid.set(ylim=(0, 1.02))
            grid.set_axis_labels(
                "Exact matched metacell count",
                "Mean score across clustering resolutions",
            )
            grid.fig.subplots_adjust(top=0.90)
            grid.fig.suptitle(
                f"{dataset_display_name} batch integration: matched-count sensitivity"
            )
            grid.fig.savefig(
                sensitivity_dir / "fig4_matched_count_clustering_sensitivity_overview.png",
                dpi=dpi,
                bbox_inches="tight",
                facecolor="white",
            )
            plt.close(grid.fig)

            individual_metric_dir = sensitivity_dir / "clustering_metrics"
            safe_mkdir(individual_metric_dir)
            for representation in metric_means["representation"].drop_duplicates():
                for metric in metric_means["metric"].drop_duplicates():
                    subset = metric_means[
                        (metric_means["representation"] == representation)
                        & (metric_means["metric"] == metric)
                    ]
                    if subset.empty:
                        continue
                    fig, ax = plt.subplots(figsize=(6.5, 5.0))
                    sns.lineplot(
                        data=subset,
                        x="target_metacells",
                        y="mean_score_across_clustering_resolutions",
                        hue="method",
                        marker="o",
                        palette=CUSTOM_PALETTE,
                        ax=ax,
                    )
                    ax.axvline(primary_target, color="0.4", linestyle="--", linewidth=0.8)
                    ax.set_ylim(0, 1.02)
                    ax.set_title(f"{metric} — {representation}")
                    ax.set_xlabel("Exact matched metacell count")
                    ax.set_ylabel("Mean score across clustering resolutions")
                    ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
                    fig.tight_layout()
                    fig.savefig(
                        individual_metric_dir
                        / f"{slugify(representation).lower()}__{slugify(metric).lower()}.png",
                        dpi=dpi,
                        bbox_inches="tight",
                        facecolor="white",
                    )
                    plt.close(fig)

    if all_lisi:
        lisi = pd.concat(all_lisi, ignore_index=True)
        save_csv(lisi, intermediate_root / "fig4_matched_count_lisi_all.csv")
        method_lisi = lisi[lisi["method"] != "Original cells"].copy()
        lisi_means = (
            method_lisi.groupby(
                ["target_metacells", "method", "metric"], as_index=False
            )["score"]
            .mean()
            .rename(columns={"score": "mean_score_across_celltypes"})
        )
        save_csv(lisi_means, intermediate_root / "fig4_matched_count_lisi_means.csv")
        if not lisi_means.empty:
            grid = sns.relplot(
                data=lisi_means,
                x="target_metacells",
                y="mean_score_across_celltypes",
                hue="method",
                col="metric",
                kind="line",
                marker="o",
                palette=CUSTOM_PALETTE,
                height=4.2,
                aspect=1.1,
                facet_kws={"sharey": True},
            )
            grid.set(ylim=(0, 1.02))
            grid.set_axis_labels("Exact matched metacell count", "Mean score")
            grid.fig.subplots_adjust(top=0.85)
            grid.fig.suptitle(
                f"{dataset_display_name} batch integration: matched-count LISI sensitivity"
            )
            grid.fig.savefig(
                sensitivity_dir / "fig4_matched_count_lisi_sensitivity_overview.png",
                dpi=dpi,
                bbox_inches="tight",
                facecolor="white",
            )
            plt.close(grid.fig)


def plot_fig5(
    intermediate_dir: Path,
    figure_dir: Path,
    dpi: int,
    fig4_primary_target: int = 1000,
) -> None:
    fig5_dir = figure_dir / "fig5_de_preservation" / "individual_panels"
    safe_mkdir(fig5_dir)
    matched_raw_umap_path = (
        intermediate_dir.parent
        / "fig4_batch_integration"
        / "matched_counts"
        / f"m{fig4_primary_target}"
        / "original_unintegrated_umap.csv.gz"
    )
    legacy_raw_umap_path = (
        intermediate_dir.parent
        / "fig4_batch_integration"
        / "original_unintegrated_umap.csv.gz"
    )
    raw_umap_path = (
        matched_raw_umap_path
        if matched_raw_umap_path.exists()
        else legacy_raw_umap_path
    )
    if raw_umap_path.exists():
        raw_umap = pd.read_csv(raw_umap_path)
        plot_umap_individuals(
            raw_umap,
            fig5_dir / "umap_original_cells",
            "Original PBMC cells",
            2.0,
            dpi,
            ["celltype", "donor", "condition"],
        )
    celltype_de = read_csvs(intermediate_dir.glob("*_celltype_de.csv.gz"))
    if not celltype_de.empty:
        celltype_de["rank"] = (
            celltype_de.groupby(["method", "group"], sort=False).cumcount() + 1
        )
        original_de = celltype_de[celltype_de["method"] == "Original cells"]
        selected_genes: List[str] = []
        for _, group_frame in original_de.groupby("group", sort=False):
            for gene in group_frame.head(5)["names"].astype(str):
                if gene not in selected_genes:
                    selected_genes.append(gene)
        matrix_rows = []
        row_names = []
        max_rank = int(celltype_de["rank"].max()) + 1
        for method in celltype_de["method"].drop_duplicates():
            method_frame = celltype_de[celltype_de["method"] == method]
            for group in original_de["group"].drop_duplicates():
                group_frame = method_frame[method_frame["group"] == group]
                rank_map = dict(
                    zip(group_frame["names"].astype(str), group_frame["rank"].astype(int))
                )
                matrix_rows.append([rank_map.get(gene, max_rank) for gene in selected_genes])
                row_names.append(f"{method} | {group}")
        if matrix_rows and selected_genes:
            heatmap_frame = pd.DataFrame(
                matrix_rows, index=row_names, columns=selected_genes
            )
            fig_width = max(10, 0.35 * len(selected_genes))
            fig_height = max(5, 0.3 * len(row_names))
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))
            sns.heatmap(
                heatmap_frame,
                cmap="mako_r",
                vmin=1,
                vmax=max_rank,
                cbar_kws={"label": "DE rank (lower is stronger)"},
                ax=ax,
            )
            ax.set_title("Cell-type DE ranks at matched 10x compression")
            ax.set_xlabel("Top genes selected from original-cell DE")
            ax.set_ylabel("")
            ax.tick_params(axis="x", rotation=60)
            fig.tight_layout()
            fig.savefig(
                fig5_dir / "fig5_celltype_de_rank_heatmap.png",
                dpi=dpi,
                bbox_inches="tight",
                facecolor="white",
            )
            plt.close(fig)

    concordance = read_csvs(intermediate_dir.glob("*_celltype_rank_concordance.csv"))
    if not concordance.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(
            data=concordance,
            x="method",
            y="kendall_tau",
            hue="method",
            palette=CUSTOM_PALETTE,
            legend=False,
            ax=ax,
        )
        sns.stripplot(
            data=concordance,
            x="method",
            y="kendall_tau",
            color="black",
            size=3,
            alpha=0.6,
            ax=ax,
        )
        ax.set_title("Cell-type DE rank preservation at matched 10x compression")
        ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        fig.savefig(
            fig5_dir / "fig5_celltype_de_kendall_tau.png",
            dpi=dpi,
            bbox_inches="tight",
            facecolor="white",
        )
        plt.close(fig)

    correlations = read_csvs(intermediate_dir.glob("*_condition_correlations.csv"))
    if not correlations.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(
            data=correlations,
            x="method",
            y="pearson_r",
            hue="method",
            palette=CUSTOM_PALETTE,
            legend=False,
            ax=ax,
        )
        sns.stripplot(
            data=correlations,
            x="method",
            y="pearson_r",
            color="black",
            size=3,
            alpha=0.6,
            ax=ax,
        )
        ax.set_title("Condition DE preservation at matched 10x compression")
        ax.tick_params(axis="x", rotation=30)
        finite_correlations = correlations["pearson_r"].replace([np.inf, -np.inf], np.nan).dropna()
        if not finite_correlations.empty:
            lower = max(-1.0, np.floor((finite_correlations.min() - 0.01) * 100) / 100)
            upper = min(1.01, max(1.0, finite_correlations.max() + 0.005))
            if upper - lower < 0.02:
                lower = upper - 0.02
            ax.set_ylim(lower, upper)
        ax.axhline(1.0, color="0.5", linestyle="--", linewidth=0.8)
        ax.set_ylabel("Pearson correlation with original-cell log fold changes")
        fig.tight_layout()
        fig.savefig(
            fig5_dir / "fig5_condition_de_pearson.png",
            dpi=dpi,
            bbox_inches="tight",
            facecolor="white",
        )
        plt.close(fig)


def plot_native_fig4_grid(
    intermediate_root: Path,
    figure_root: Path,
    targets: Sequence[int],
    dpi: int,
) -> None:
    """Plot native-resolution Fig. 4 results against realized counts."""
    safe_mkdir(figure_root)
    all_metrics: List[pd.DataFrame] = []
    all_lisi: List[pd.DataFrame] = []
    for requested in [int(value) for value in targets]:
        target_dir = intermediate_root / f"requested_m{requested}"
        if not target_dir.is_dir():
            logger.warning("No native Fig. 4 checkpoint directory for requested m=%d", requested)
            continue
        plot_fig4(target_dir, figure_root / f"requested_m{requested}", dpi)
        metrics = read_csvs(target_dir.glob("*_clustering_metrics.csv"))
        if not metrics.empty:
            if "requested_metacells" not in metrics:
                metrics["requested_metacells"] = requested
            all_metrics.append(metrics)
        lisi = read_csvs(target_dir.glob("*_lisi_summary.csv"))
        if not lisi.empty:
            if "requested_metacells" not in lisi:
                lisi["requested_metacells"] = requested
            all_lisi.append(lisi)

    sensitivity_dir = figure_root / "native_resolution_sensitivity"
    safe_mkdir(sensitivity_dir)
    if all_metrics:
        metrics = pd.concat(all_metrics, ignore_index=True)
        save_csv(metrics, intermediate_root / "fig4_native_clustering_metrics_all.csv")
        method_metrics = metrics[metrics["method"] != "Original cells"].copy()
        means = (
            method_metrics.groupby(
                [
                    "requested_metacells",
                    "realized_metacells",
                    "method",
                    "representation",
                    "metric",
                ],
                as_index=False,
            )["score"]
            .mean()
            .rename(columns={"score": "mean_score_across_clustering_resolutions"})
        )
        save_csv(means, intermediate_root / "fig4_native_clustering_metric_means.csv")
        if not means.empty:
            grid = sns.relplot(
                data=means,
                x="realized_metacells",
                y="mean_score_across_clustering_resolutions",
                hue="method",
                row="representation",
                col="metric",
                kind="line",
                marker="o",
                palette=CUSTOM_PALETTE,
                height=3.6,
                aspect=1.05,
                facet_kws={"sharey": True, "sharex": True},
            )
            grid.set(ylim=(0, 1.02))
            grid.set_axis_labels(
                "Realized native metacell count",
                "Mean score across clustering resolutions",
            )
            grid.fig.subplots_adjust(top=0.90)
            grid.fig.suptitle("PBMC batch integration: native-resolution sensitivity")
            grid.fig.savefig(
                sensitivity_dir / "fig4_native_clustering_sensitivity_overview.png",
                dpi=dpi,
                bbox_inches="tight",
                facecolor="white",
            )
            plt.close(grid.fig)

    if all_lisi:
        lisi = pd.concat(all_lisi, ignore_index=True)
        save_csv(lisi, intermediate_root / "fig4_native_lisi_all.csv")
        method_lisi = lisi[lisi["method"] != "Original cells"].copy()
        means = (
            method_lisi.groupby(
                ["requested_metacells", "realized_metacells", "method", "metric"],
                as_index=False,
            )["score"]
            .mean()
            .rename(columns={"score": "mean_score_across_celltypes"})
        )
        save_csv(means, intermediate_root / "fig4_native_lisi_means.csv")
        if not means.empty:
            grid = sns.relplot(
                data=means,
                x="realized_metacells",
                y="mean_score_across_celltypes",
                hue="method",
                col="metric",
                kind="line",
                marker="o",
                palette=CUSTOM_PALETTE,
                height=4.2,
                aspect=1.1,
            )
            grid.set(ylim=(0, 1.02))
            grid.set_axis_labels("Realized native metacell count", "Mean score")
            grid.fig.subplots_adjust(top=0.84)
            grid.fig.suptitle("PBMC batch integration: native-resolution LISI")
            grid.fig.savefig(
                sensitivity_dir / "fig4_native_lisi_sensitivity_overview.png",
                dpi=dpi,
                bbox_inches="tight",
                facecolor="white",
            )
            plt.close(grid.fig)


def plot_native_fig5_grid(
    intermediate_root: Path,
    figure_root: Path,
    targets: Sequence[int],
    dpi: int,
) -> None:
    """Plot per-resolution DE preservation and cross-resolution sensitivity."""
    safe_mkdir(figure_root)
    all_concordance: List[pd.DataFrame] = []
    all_correlations: List[pd.DataFrame] = []
    all_summaries: List[pd.DataFrame] = []
    for requested in [int(value) for value in targets]:
        target_dir = intermediate_root / f"requested_m{requested}"
        if not target_dir.is_dir():
            logger.warning("No native Fig. 5 checkpoint directory for requested m=%d", requested)
            continue
        panel_dir = figure_root / f"requested_m{requested}" / "individual_panels"
        safe_mkdir(panel_dir)
        concordance = read_csvs(target_dir.glob("*_celltype_rank_concordance.csv"))
        correlations = read_csvs(target_dir.glob("*_condition_correlations.csv"))
        summary_path = target_dir / "metacell_summary.csv"
        if summary_path.is_file():
            summary = pd.read_csv(summary_path)
            all_summaries.append(summary)
        if not concordance.empty:
            all_concordance.append(concordance)
            fig, ax = plt.subplots(figsize=(8.5, 5.2))
            sns.boxplot(
                data=concordance,
                x="method",
                y="kendall_tau",
                hue="method",
                palette=CUSTOM_PALETTE,
                legend=False,
                ax=ax,
            )
            sns.stripplot(
                data=concordance,
                x="method",
                y="kendall_tau",
                color="black",
                size=3,
                alpha=0.55,
                ax=ax,
            )
            ax.set_title(
                f"Cell-type DE rank preservation — native grid requested m={requested}"
            )
            ax.tick_params(axis="x", rotation=30)
            fig.tight_layout()
            fig.savefig(
                panel_dir / "fig5_celltype_de_kendall_tau.png",
                dpi=dpi,
                bbox_inches="tight",
                facecolor="white",
            )
            plt.close(fig)
        if not correlations.empty:
            all_correlations.append(correlations)
            fig, ax = plt.subplots(figsize=(8.5, 5.2))
            sns.boxplot(
                data=correlations,
                x="method",
                y="pearson_r",
                hue="method",
                palette=CUSTOM_PALETTE,
                legend=False,
                ax=ax,
            )
            sns.stripplot(
                data=correlations,
                x="method",
                y="pearson_r",
                color="black",
                size=3,
                alpha=0.55,
                ax=ax,
            )
            ax.axhline(1.0, color="0.5", linestyle="--", linewidth=0.8)
            ax.set_title(
                f"Condition DE preservation — native grid requested m={requested}"
            )
            ax.set_ylabel("Pearson correlation with original-cell log fold changes")
            ax.tick_params(axis="x", rotation=30)
            fig.tight_layout()
            fig.savefig(
                panel_dir / "fig5_condition_de_pearson.png",
                dpi=dpi,
                bbox_inches="tight",
                facecolor="white",
            )
            plt.close(fig)

    sensitivity_dir = figure_root / "native_resolution_sensitivity"
    safe_mkdir(sensitivity_dir)
    if all_summaries:
        save_csv(
            pd.concat(all_summaries, ignore_index=True),
            intermediate_root / "fig5_native_metacell_summary_all.csv",
        )
    if all_concordance:
        concordance = pd.concat(all_concordance, ignore_index=True)
        save_csv(concordance, intermediate_root / "fig5_native_rank_concordance_all.csv")
        means = (
            concordance.groupby(
                [
                    "method",
                    "requested_global_metacells",
                    "realized_stratified_profiles",
                ],
                as_index=False,
            )["kendall_tau"]
            .mean()
            .rename(columns={"kendall_tau": "mean_kendall_tau_across_celltypes"})
        )
        save_csv(means, intermediate_root / "fig5_native_rank_concordance_means.csv")
        fig, ax = plt.subplots(figsize=(7.4, 5.4))
        sns.lineplot(
            data=means,
            x="realized_stratified_profiles",
            y="mean_kendall_tau_across_celltypes",
            hue="method",
            marker="o",
            palette=CUSTOM_PALETTE,
            ax=ax,
        )
        ax.set_xlabel("Realized profiles after native partition × DE-stratum intersection")
        ax.set_ylabel("Mean Kendall tau")
        ax.set_title("Cell-type DE preservation across native resolutions")
        ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
        fig.tight_layout()
        fig.savefig(
            sensitivity_dir / "fig5_native_celltype_de_sensitivity.png",
            dpi=dpi,
            bbox_inches="tight",
            facecolor="white",
        )
        plt.close(fig)
    if all_correlations:
        correlations = pd.concat(all_correlations, ignore_index=True)
        save_csv(correlations, intermediate_root / "fig5_native_condition_correlations_all.csv")
        means = (
            correlations.groupby(
                [
                    "method",
                    "requested_global_metacells",
                    "realized_stratified_profiles",
                ],
                as_index=False,
            )["pearson_r"]
            .mean()
            .rename(columns={"pearson_r": "mean_pearson_r_across_contrasts"})
        )
        save_csv(means, intermediate_root / "fig5_native_condition_correlation_means.csv")
        fig, ax = plt.subplots(figsize=(7.4, 5.4))
        sns.lineplot(
            data=means,
            x="realized_stratified_profiles",
            y="mean_pearson_r_across_contrasts",
            hue="method",
            marker="o",
            palette=CUSTOM_PALETTE,
            ax=ax,
        )
        ax.set_xlabel("Realized profiles after native partition × DE-stratum intersection")
        ax.set_ylabel("Mean Pearson r")
        ax.set_title("Condition DE preservation across native resolutions")
        ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
        fig.tight_layout()
        fig.savefig(
            sensitivity_dir / "fig5_native_condition_de_sensitivity.png",
            dpi=dpi,
            bbox_inches="tight",
            facecolor="white",
        )
        plt.close(fig)


def save_matched_method_legend(
    methods: Sequence[str],
    output_path: Path,
    dpi: int,
) -> None:
    """Save the shared matched-DE method legend as a tightly cropped artifact."""
    methods = [method for method in ALL_METHOD_ORDER if method in set(methods)]
    if not methods:
        return
    handles = [
        Line2D(
            [0],
            [0],
            color=CUSTOM_PALETTE[method],
            marker="o",
            markersize=5,
            linewidth=1.8,
            label=method,
        )
        for method in methods
    ]
    # Keep all methods on one horizontal row so the legend can be positioned
    # independently beneath a multi-panel figure in the manuscript.
    # Use a figure-level legend without an axes.  This guarantees one row and
    # lets bbox_inches="tight" crop to the legend itself instead of retaining
    # the otherwise empty axes rectangle.
    fig = plt.figure(figsize=(11.8, 0.42))
    legend = fig.legend(
        handles=handles,
        labels=methods,
        loc="center",
        ncol=len(methods),
        frameon=False,
        handlelength=1.6,
        handletextpad=0.35,
        columnspacing=0.85,
        borderaxespad=0,
        fontsize=8,
    )
    safe_mkdir(output_path.parent)
    fig.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
        bbox_extra_artists=(legend,),
        pad_inches=0.005,
        facecolor="white",
    )
    fig.savefig(
        output_path.with_suffix(".pdf"),
        bbox_inches="tight",
        bbox_extra_artists=(legend,),
        pad_inches=0.005,
        facecolor="white",
    )
    plt.close(fig)
    logger.info("Saved separate method legend: %s", output_path)


def plot_matched_fig5_grid(
    intermediate_root: Path,
    figure_root: Path,
    reduction_rates: Sequence[float],
    primary_reduction_rate: float,
    dpi: int,
) -> None:
    """Plot exact matched-compression DE results and three-rate sensitivity."""
    safe_mkdir(figure_root)
    all_concordance: List[pd.DataFrame] = []
    all_correlations: List[pd.DataFrame] = []
    all_summaries: List[pd.DataFrame] = []
    for reduction_rate in [float(value) for value in reduction_rates]:
        rate_name = reduction_rate_slug(reduction_rate)
        target_dir = intermediate_root / rate_name
        if not target_dir.is_dir():
            logger.warning("No matched Fig. 5 checkpoint directory for %.3gx", reduction_rate)
            continue
        panel_dir = figure_root / rate_name / "individual_panels"
        safe_mkdir(panel_dir)
        concordance = read_csvs(target_dir.glob("*_celltype_rank_concordance.csv"))
        correlations = read_csvs(target_dir.glob("*_condition_correlations.csv"))
        summary_path = target_dir / "metacell_summary.csv"
        if summary_path.is_file():
            all_summaries.append(pd.read_csv(summary_path))
        for frame in [concordance, correlations]:
            if not frame.empty and "requested_de_reduction_rate" not in frame:
                frame["requested_de_reduction_rate"] = reduction_rate

        if not concordance.empty:
            all_concordance.append(concordance)
            order = [
                method
                for method in ALL_METHOD_ORDER
                if method in set(concordance["method"].astype(str))
            ]
            fig, ax = plt.subplots(figsize=(9.0, 5.4))
            sns.boxplot(
                data=concordance,
                x="method",
                y="kendall_tau",
                order=order,
                hue="method",
                hue_order=order,
                palette=CUSTOM_PALETTE,
                legend=False,
                ax=ax,
            )
            sns.stripplot(
                data=concordance,
                x="method",
                y="kendall_tau",
                order=order,
                color="black",
                size=3,
                alpha=0.55,
                ax=ax,
            )
            ax.set_title(
                f"Cell-type DE rank preservation — exact matched {reduction_rate:g}x"
            )
            ax.set_xlabel("")
            ax.set_ylabel("Kendall tau with original-cell DE ranks")
            ax.tick_params(axis="x", rotation=30)
            fig.tight_layout()
            save_publication_figure(
                fig, panel_dir / "fig5_celltype_de_kendall_tau.png", dpi
            )

        if not correlations.empty:
            all_correlations.append(correlations)
            order = [
                method
                for method in ALL_METHOD_ORDER
                if method in set(correlations["method"].astype(str))
            ]
            fig, ax = plt.subplots(figsize=(9.0, 5.4))
            sns.boxplot(
                data=correlations,
                x="method",
                y="pearson_r",
                order=order,
                hue="method",
                hue_order=order,
                palette=CUSTOM_PALETTE,
                legend=False,
                ax=ax,
            )
            sns.stripplot(
                data=correlations,
                x="method",
                y="pearson_r",
                order=order,
                color="black",
                size=3,
                alpha=0.55,
                ax=ax,
            )
            ax.axhline(1.0, color="0.5", linestyle="--", linewidth=0.8)
            ax.set_title(
                f"Condition DE preservation — exact matched {reduction_rate:g}x"
            )
            ax.set_xlabel("")
            ax.set_ylabel("Pearson r with original-cell log fold changes")
            ax.tick_params(axis="x", rotation=30)
            fig.tight_layout()
            save_publication_figure(
                fig, panel_dir / "fig5_condition_de_pearson.png", dpi
            )

    sensitivity_dir = figure_root / "matched_compression_sensitivity"
    safe_mkdir(sensitivity_dir)
    if all_summaries:
        summaries = pd.concat(all_summaries, ignore_index=True)
        save_csv(summaries, intermediate_root / "fig5_matched_metacell_summary_all.csv")

    rank_rows: List[pd.DataFrame] = []
    if all_concordance:
        concordance_all = pd.concat(all_concordance, ignore_index=True)
        save_csv(
            concordance_all,
            intermediate_root / "fig5_matched_rank_concordance_all.csv",
        )
        rank_means = (
            concordance_all.groupby(
                ["requested_de_reduction_rate", "method"], as_index=False
            ).agg(
                mean_score=("kendall_tau", "mean"),
                std_score=("kendall_tau", "std"),
                n_comparisons=("kendall_tau", "count"),
            )
        )
        rank_means["metric"] = "celltype_de_kendall_tau"
        rank_means["rank"] = rank_means.groupby(
            "requested_de_reduction_rate"
        )["mean_score"].rank(method="min", ascending=False)
        rank_rows.append(rank_means)
        fig, ax = plt.subplots(figsize=(7.8, 5.4))
        sns.lineplot(
            data=rank_means,
            x="requested_de_reduction_rate",
            y="mean_score",
            hue="method",
            marker="o",
            palette=CUSTOM_PALETTE,
            legend=False,
            ax=ax,
        )
        ax.axvline(primary_reduction_rate, color="0.4", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Exact within-stratum compression (fold)")
        ax.set_ylabel("Mean Kendall tau across cell types")
        ax.set_title("Cell-type DE preservation across matched compression rates")
        fig.tight_layout()
        save_publication_figure(
            fig, sensitivity_dir / "fig5_matched_celltype_de_sensitivity.png", dpi
        )

    if all_correlations:
        correlations_all = pd.concat(all_correlations, ignore_index=True)
        save_csv(
            correlations_all,
            intermediate_root / "fig5_matched_condition_correlations_all.csv",
        )
        correlation_means = (
            correlations_all.groupby(
                ["requested_de_reduction_rate", "method"], as_index=False
            ).agg(
                mean_score=("pearson_r", "mean"),
                std_score=("pearson_r", "std"),
                n_comparisons=("pearson_r", "count"),
            )
        )
        correlation_means["metric"] = "condition_de_pearson_r"
        correlation_means["rank"] = correlation_means.groupby(
            "requested_de_reduction_rate"
        )["mean_score"].rank(method="min", ascending=False)
        rank_rows.append(correlation_means)
        fig, ax = plt.subplots(figsize=(7.8, 5.4))
        sns.lineplot(
            data=correlation_means,
            x="requested_de_reduction_rate",
            y="mean_score",
            hue="method",
            marker="o",
            palette=CUSTOM_PALETTE,
            legend=False,
            ax=ax,
        )
        ax.axvline(primary_reduction_rate, color="0.4", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Exact within-stratum compression (fold)")
        ax.set_ylabel("Mean Pearson r across condition contrasts")
        ax.set_title("Condition DE preservation across matched compression rates")
        fig.tight_layout()
        save_publication_figure(
            fig, sensitivity_dir / "fig5_matched_condition_de_sensitivity.png", dpi
        )

    if rank_rows:
        rank_summary = pd.concat(rank_rows, ignore_index=True)
        save_csv(rank_summary, intermediate_root / "fig5_matched_rank_summary.csv")
        overall = (
            rank_summary.groupby("method", as_index=False).agg(
                mean_rank=("rank", "mean"),
                std_rank=("rank", "std"),
                n_metric_rate_combinations=("rank", "count"),
            )
            .sort_values(["mean_rank", "method"])
        )
        save_csv(overall, intermediate_root / "fig5_matched_overall_rank_summary.csv")
        save_matched_method_legend(
            rank_summary["method"].drop_duplicates().astype(str).tolist(),
            sensitivity_dir / "fig5_matched_methods_legend.png",
            dpi,
        )


# =========================================================
# MetaQ-paper-style individual panels
# =========================================================
def save_publication_figure(fig: plt.Figure, png_path: Path, dpi: int) -> None:
    """Save a high-DPI PNG plus an editable vector PDF for one panel."""
    safe_mkdir(png_path.parent)
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(png_path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("Saved publication panel: %s", png_path)


def category_palette(values: Sequence[object], kind: str = "husl") -> Dict[str, object]:
    categories = sorted({str(value) for value in values})
    colors = sns.color_palette(kind, n_colors=max(1, len(categories)))
    return dict(zip(categories, colors))


def plot_metaq_style_umap(
    frame: pd.DataFrame,
    color_key: str,
    title: str,
    output_path: Path,
    point_size: float,
    dpi: int,
    palette: Optional[Mapping[str, object]] = None,
) -> None:
    """Draw a compact square UMAP panel matching the layout of MetaQ Figs. 4a-d/5a."""
    if frame.empty or color_key not in frame:
        return
    work = frame.copy()
    work[color_key] = work[color_key].astype(str)
    if palette is None:
        palette = category_palette(work[color_key], "husl")
    fig, ax = plt.subplots(figsize=(4.2, 4.0))
    sns.scatterplot(
        data=work,
        x="UMAP1",
        y="UMAP2",
        hue=color_key,
        palette=palette,
        s=point_size,
        linewidth=0,
        alpha=0.80,
        rasterized=True,
        ax=ax,
    )
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("UMAP1", fontsize=8)
    ax.set_ylabel("UMAP2", fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    if ax.legend_ is not None:
        ax.legend(
            title=color_key.replace("celltype", "Cell Type").title(),
            bbox_to_anchor=(1.01, 0.5),
            loc="center left",
            frameon=False,
            fontsize=6.5,
            title_fontsize=7,
            markerscale=1.8,
            borderaxespad=0,
        )
    sns.despine(ax=ax, left=False, bottom=False)
    fig.tight_layout()
    save_publication_figure(fig, output_path, dpi)


def paper_comparison_palette(methods: Sequence[str], focal_method: Optional[str]) -> Dict[str, str]:
    del focal_method  # Focal status is conveyed by the panel title, not recoloring.
    return {
        method: METAQ_PAPER_COMPARATOR_PALETTE.get(
            method, CUSTOM_PALETTE.get(method, "#777777")
        )
        for method in methods
    }


def plot_metaq_grouped_clustering_bars(
    metrics: pd.DataFrame,
    methods: Sequence[str],
    focal_method: Optional[str],
    representation: str,
    include_original_harmony: bool,
    title: str,
    output_path: Path,
    dpi: int,
) -> pd.DataFrame:
    """Reproduce MetaQ Fig. 4g/h: mean bars with resolution scores as black dots."""
    rows: List[pd.DataFrame] = []
    if include_original_harmony:
        original = metrics[
            (metrics["method"] == "Original cells")
            & (metrics["representation"] == "integrated_cells")
        ].copy()
        if not original.empty:
            original["plot_method"] = "Harmony"
            rows.append(original)
    for method in methods:
        subset = metrics[
            (metrics["method"] == method)
            & (metrics["representation"] == representation)
        ].copy()
        if not subset.empty:
            subset["plot_method"] = method
            rows.append(subset)
    if not rows:
        return pd.DataFrame()
    data = pd.concat(rows, ignore_index=True)
    metric_order = ["AMI", "ARI", "Homogeneity"]
    method_order = (["Harmony"] if include_original_harmony else []) + [
        method for method in methods if method in set(data["plot_method"])
    ]
    palette = paper_comparison_palette(method_order, focal_method)
    if "Harmony" in method_order:
        palette["Harmony"] = "#4C7899"

    fig_width = max(7.3, 1.02 * len(method_order) + 2.4)
    fig, ax = plt.subplots(figsize=(fig_width, 4.1))
    x = np.arange(len(metric_order), dtype=float)
    width = 0.80 / max(1, len(method_order))
    for method_idx, method in enumerate(method_order):
        centers = x - 0.40 + width / 2 + method_idx * width
        method_frame = data[data["plot_method"] == method]
        means = []
        for metric_idx, metric in enumerate(metric_order):
            values = (
                method_frame.loc[method_frame["metric"] == metric, "score"]
                .astype(float)
                .dropna()
                .to_numpy()
            )
            means.append(float(np.mean(values)) if len(values) else np.nan)
            if len(values):
                jitter = np.linspace(-0.16 * width, 0.16 * width, len(values))
                ax.scatter(
                    np.full(len(values), centers[metric_idx]) + jitter,
                    values,
                    s=8,
                    facecolors="black",
                    edgecolors="white",
                    linewidths=0.25,
                    zorder=4,
                )
        ax.bar(
            centers,
            means,
            width=width * 0.94,
            color=palette[method],
            label=method if method == "Harmony" else f"+ {method}",
            edgecolor="white",
            linewidth=0.35,
            zorder=2,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(metric_order)
    ax.set_ylabel("Score")
    ax.set_title(title, fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.legend(
        frameon=False,
        bbox_to_anchor=(1.01, 0.5),
        loc="center left",
        fontsize=7.2,
        ncol=1,
    )
    sns.despine(ax=ax)
    fig.tight_layout()
    save_publication_figure(fig, output_path, dpi)
    data["panel"] = output_path.stem
    return data


def significance_stars(p_value: float) -> str:
    if not np.isfinite(p_value):
        return "ns"
    if p_value <= 1e-4:
        return "****"
    if p_value <= 1e-3:
        return "***"
    if p_value <= 1e-2:
        return "**"
    if p_value <= 5e-2:
        return "*"
    return "ns"


def format_p_value(p_value: float) -> str:
    """Format an exact p-value compactly enough for a publication bracket."""
    if not np.isfinite(p_value):
        return "NA"
    if p_value < 1e-3:
        return f"{p_value:.2e}"
    return f"{p_value:.3f}"


def focal_ttests(
    frame: pd.DataFrame,
    value_key: str,
    focal_method: str,
    comparators: Sequence[str],
) -> pd.DataFrame:
    """Paper-matched independent two-sided T-tests for bracket annotations."""
    focal = (
        frame.loc[frame["method"] == focal_method, value_key]
        .astype(float)
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .to_numpy()
    )
    rows = []
    for comparator in comparators:
        other = (
            frame.loc[frame["method"] == comparator, value_key]
            .astype(float)
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
            .to_numpy()
        )
        statistic, p_value = ttest_ind(focal, other, equal_var=True, nan_policy="omit")
        rows.append(
            {
                "focal_method": focal_method,
                "comparator": comparator,
                "test": "two-sided independent T-test (MetaQ paper convention)",
                "n_focal": len(focal),
                "n_comparator": len(other),
                "t_statistic": statistic,
                "p_value": p_value,
                "stars": significance_stars(float(p_value)),
                "mean_focal": float(np.mean(focal)) if len(focal) else np.nan,
                "mean_comparator": float(np.mean(other)) if len(other) else np.nan,
                "mean_difference": (
                    float(np.mean(focal) - np.mean(other))
                    if len(focal) and len(other)
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def camp_vs_baseline_ttests(
    frame: pd.DataFrame,
    value_key: str,
    method_order: Sequence[str],
) -> pd.DataFrame:
    """Return the complete CAMP1-4 versus baseline comparison table."""
    camp_methods = [
        method for method in method_order if str(method).upper().startswith("CAMP")
    ]
    baseline_methods = [
        method for method in METAQ_PAPER_BASELINES if method in method_order
    ]
    tables = [
        focal_ttests(frame, value_key, focal_method, baseline_methods)
        for focal_method in camp_methods
    ]
    tables = [table for table in tables if not table.empty]
    if not tables:
        return pd.DataFrame()
    result = pd.concat(tables, ignore_index=True)
    p_values = result["p_value"].to_numpy(dtype=float)
    adjusted = np.full(len(p_values), np.nan, dtype=float)
    finite_indices = np.flatnonzero(np.isfinite(p_values))
    if len(finite_indices):
        ranked_indices = finite_indices[np.argsort(p_values[finite_indices])]
        ranked_p = p_values[ranked_indices]
        raw_adjusted = ranked_p * len(ranked_p) / np.arange(1, len(ranked_p) + 1)
        monotone_adjusted = np.minimum.accumulate(raw_adjusted[::-1])[::-1]
        adjusted[ranked_indices] = np.minimum(monotone_adjusted, 1.0)
    result["p_value_bh"] = adjusted
    result["stars_bh"] = [significance_stars(value) for value in adjusted]
    return result


def add_vertical_significance_brackets(
    ax: plt.Axes,
    order: Sequence[str],
    tests: pd.DataFrame,
    focal_method: str,
) -> None:
    y_min, y_max = ax.get_ylim()
    span = max(1e-6, y_max - y_min)
    base = y_max + 0.025 * span
    step = 0.075 * span
    focal_x = order.index(focal_method)
    for bracket_idx, row in tests.iterrows():
        comparator = str(row["comparator"])
        if comparator not in order:
            continue
        y = base + bracket_idx * step
        other_x = order.index(comparator)
        ax.plot(
            [focal_x, focal_x, other_x, other_x],
            [y, y + 0.014 * span, y + 0.014 * span, y],
            color="black",
            linewidth=0.65,
            clip_on=False,
        )
        label = f"p={format_p_value(float(row['p_value']))} ({row['stars']})"
        ax.text(
            (focal_x + other_x) / 2,
            y + 0.018 * span,
            label,
            ha="center",
            va="bottom",
            fontsize=6.5,
            clip_on=False,
        )
    ax.set_ylim(y_min, base + max(1, len(tests)) * step + 0.04 * span)


def add_horizontal_significance_brackets(
    ax: plt.Axes,
    order: Sequence[str],
    tests: pd.DataFrame,
    focal_method: str,
) -> None:
    x_min, x_max = ax.get_xlim()
    span = max(1e-6, x_max - x_min)
    base = x_max + 0.02 * span
    step = 0.075 * span
    focal_y = order.index(focal_method)
    for bracket_idx, row in tests.iterrows():
        comparator = str(row["comparator"])
        if comparator not in order:
            continue
        x = base + bracket_idx * step
        other_y = order.index(comparator)
        ax.plot(
            [x, x + 0.012 * span, x + 0.012 * span, x],
            [focal_y, focal_y, other_y, other_y],
            color="black",
            linewidth=0.65,
            clip_on=False,
        )
        ax.text(
            x + 0.016 * span,
            (focal_y + other_y) / 2,
            f"p={format_p_value(float(row['p_value']))} ({row['stars']})",
            ha="left",
            va="center",
            fontsize=6.5,
            clip_on=False,
        )
    ax.set_xlim(x_min, base + max(1, len(tests)) * step + 0.06 * span)


def plot_metaq_sankey(
    counts: pd.DataFrame,
    title: str,
    output_path: Path,
    dpi: int,
    max_celltypes: Optional[int] = None,
) -> None:
    """Draw a compact cell-type-to-cluster alluvial panel without extra packages."""
    if counts.empty:
        return
    data = counts.copy()
    if max_celltypes is not None:
        totals = data.groupby("celltype", as_index=False)["n_cells"].sum()
        keep = totals.nlargest(max_celltypes, "n_cells")["celltype"].astype(str).tolist()
        data["celltype"] = data["celltype"].astype(str).where(
            data["celltype"].astype(str).isin(keep), "Other"
        )
    data["cluster"] = "Cluster " + data["cluster"].astype(str)
    data = data.groupby(["celltype", "cluster"], as_index=False)["n_cells"].sum()
    left_order = (
        data.groupby("celltype")["n_cells"].sum().sort_values(ascending=False).index.tolist()
    )
    right_order = (
        data.groupby("cluster")["n_cells"].sum().sort_values(ascending=False).index.tolist()
    )
    total = float(data["n_cells"].sum())
    gap = min(0.006, 0.16 / max(1, max(len(left_order), len(right_order))))

    def intervals(labels: Sequence[str], key: str) -> Dict[str, Tuple[float, float]]:
        usable = 1.0 - gap * max(0, len(labels) - 1)
        cursor = 0.0
        result: Dict[str, Tuple[float, float]] = {}
        for label in labels:
            amount = float(data.loc[data[key] == label, "n_cells"].sum())
            height = usable * amount / total
            result[label] = (cursor, cursor + height)
            cursor += height + gap
        return result

    left_bounds = intervals(left_order, "celltype")
    right_bounds = intervals(right_order, "cluster")
    left_cursor = {label: bounds[0] for label, bounds in left_bounds.items()}
    right_cursor = {label: bounds[0] for label, bounds in right_bounds.items()}
    left_scale = {
        label: (bounds[1] - bounds[0])
        / float(data.loc[data["celltype"] == label, "n_cells"].sum())
        for label, bounds in left_bounds.items()
    }
    right_scale = {
        label: (bounds[1] - bounds[0])
        / float(data.loc[data["cluster"] == label, "n_cells"].sum())
        for label, bounds in right_bounds.items()
    }
    colors = category_palette(left_order, "husl")
    fig_height = max(4.8, 0.25 * len(left_order))
    fig, ax = plt.subplots(figsize=(4.2, fig_height))

    def spread_label_positions(
        bounds: Mapping[str, Tuple[float, float]], minimum_gap: float
    ) -> Dict[str, float]:
        """Keep labels legible while retaining every cell type in the Sankey."""
        ordered = sorted(bounds, key=lambda label: np.mean(bounds[label]))
        positions = {
            label: float(np.mean(bounds[label])) for label in ordered
        }
        for label_idx in range(1, len(ordered)):
            previous = ordered[label_idx - 1]
            current = ordered[label_idx]
            positions[current] = max(
                positions[current], positions[previous] + minimum_gap
            )
        overflow = max(0.0, positions[ordered[-1]] - 0.96) if ordered else 0.0
        if overflow:
            for label in ordered:
                positions[label] -= overflow
        for label_idx in range(len(ordered) - 2, -1, -1):
            current = ordered[label_idx]
            following = ordered[label_idx + 1]
            positions[current] = min(
                positions[current], positions[following] - minimum_gap
            )
        underflow = max(0.0, 0.02 - positions[ordered[0]]) if ordered else 0.0
        if underflow:
            for label in ordered:
                positions[label] += underflow
        return positions

    left_text_y = spread_label_positions(left_bounds, minimum_gap=0.031)
    right_text_y = spread_label_positions(right_bounds, minimum_gap=0.031)
    for _, row in data.sort_values(["celltype", "cluster"]).iterrows():
        celltype = str(row["celltype"])
        cluster = str(row["cluster"])
        amount = float(row["n_cells"])
        left_y0 = left_cursor[celltype]
        left_y1 = left_y0 + amount * left_scale[celltype]
        right_y0 = right_cursor[cluster]
        right_y1 = right_y0 + amount * right_scale[cluster]
        left_cursor[celltype] = left_y1
        right_cursor[cluster] = right_y1
        vertices = [
            (0.20, left_y0),
            (0.46, left_y0),
            (0.54, right_y0),
            (0.80, right_y0),
            (0.80, right_y1),
            (0.54, right_y1),
            (0.46, left_y1),
            (0.20, left_y1),
            (0.20, left_y0),
        ]
        codes = [
            MplPath.MOVETO,
            MplPath.CURVE4,
            MplPath.CURVE4,
            MplPath.CURVE4,
            MplPath.LINETO,
            MplPath.CURVE4,
            MplPath.CURVE4,
            MplPath.CURVE4,
            MplPath.CLOSEPOLY,
        ]
        ax.add_patch(
            PathPatch(
                MplPath(vertices, codes),
                facecolor=colors[celltype],
                edgecolor="none",
                alpha=0.55,
            )
        )
    for celltype, (y0, y1) in left_bounds.items():
        ax.add_patch(Rectangle((0.17, y0), 0.03, y1 - y0, color=colors[celltype]))
        center = (y0 + y1) / 2
        ax.plot([0.17, 0.145], [center, left_text_y[celltype]], color="#777777", lw=0.35)
        ax.text(0.14, left_text_y[celltype], celltype, ha="right", va="center", fontsize=6.5)
    for cluster, (y0, y1) in right_bounds.items():
        ax.add_patch(Rectangle((0.80, y0), 0.03, y1 - y0, color="#777777"))
        center = (y0 + y1) / 2
        ax.plot([0.83, 0.855], [center, right_text_y[cluster]], color="#777777", lw=0.35)
        ax.text(0.86, right_text_y[cluster], cluster, ha="left", va="center", fontsize=6.5)
    ax.set_xlim(0.02, 0.98)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title(title, fontsize=10, y=1.025)
    fig.tight_layout()
    save_publication_figure(fig, output_path, dpi)


def plot_metaq_marker_panel(
    frame: pd.DataFrame,
    marker_gene: str,
    marker_celltype: str,
    inset_batch: str,
    title: str,
    output_path: Path,
    dpi: int,
    vmax: float,
) -> None:
    subset = frame[frame["celltype"].astype(str) == marker_celltype].copy()
    if subset.empty:
        return
    fig, ax = plt.subplots(figsize=(4.4, 4.0))
    points = ax.scatter(
        subset["UMAP1"],
        subset["UMAP2"],
        c=subset["expression"],
        cmap="coolwarm",
        vmin=0,
        vmax=vmax,
        s=3.5,
        linewidths=0,
        rasterized=True,
    )
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("UMAP1", fontsize=8)
    ax.set_ylabel("UMAP2", fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    inset = ax.inset_axes([0.56, 0.58, 0.38, 0.34])
    inset_data = subset[subset["batch"].astype(str) == str(inset_batch)]
    inset.scatter(
        inset_data["UMAP1"],
        inset_data["UMAP2"],
        c=inset_data["expression"],
        cmap="coolwarm",
        vmin=0,
        vmax=vmax,
        s=3.5,
        linewidths=0,
        rasterized=True,
    )
    inset.set_xticks([])
    inset.set_yticks([])
    inset.set_title(f"In batch {inset_batch}", fontsize=6.5)
    colorbar = fig.colorbar(points, ax=ax, fraction=0.045, pad=0.02)
    colorbar.set_label(f"{marker_gene} expression", fontsize=7)
    colorbar.ax.tick_params(labelsize=6)
    fig.tight_layout()
    save_publication_figure(fig, output_path, dpi)


def plot_metaq_marker_pair(
    marker_data: pd.DataFrame,
    focal_method: str,
    marker_gene: str,
    marker_celltype: str,
    inset_batch: str,
    output_path: Path,
    dpi: int,
    vmax: float,
) -> None:
    """Match the focal-method-versus-Harmony arrangement in MetaQ Fig. 4f."""
    methods = [focal_method, "Harmony"]
    if not set(methods).issubset(set(marker_data["method"].astype(str))):
        return
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.8), sharex=False, sharey=False)
    last_points = None
    for panel_idx, (ax, method) in enumerate(zip(axes, methods)):
        subset = marker_data[
            (marker_data["method"].astype(str) == method)
            & (marker_data["celltype"].astype(str) == marker_celltype)
        ]
        last_points = ax.scatter(
            subset["UMAP1"],
            subset["UMAP2"],
            c=subset["expression"],
            cmap="coolwarm",
            vmin=0,
            vmax=vmax,
            s=3.5,
            linewidths=0,
            rasterized=True,
        )
        method_title = f"Harmony + {method}" if method != "Harmony" else "Harmony"
        ax.set_title(f"{marker_gene} ({method_title})", fontsize=9)
        ax.set_xlabel("UMAP1", fontsize=8)
        ax.set_ylabel("UMAP2", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        if panel_idx == 0:
            inset = ax.inset_axes([0.56, 0.58, 0.38, 0.34])
            inset_data = subset[subset["batch"].astype(str) == str(inset_batch)]
            inset.scatter(
                inset_data["UMAP1"],
                inset_data["UMAP2"],
                c=inset_data["expression"],
                cmap="coolwarm",
                vmin=0,
                vmax=vmax,
                s=3.5,
                linewidths=0,
                rasterized=True,
            )
            inset.set_xticks([])
            inset.set_yticks([])
            inset.set_title(f"In batch {inset_batch}", fontsize=6.5)
    if last_points is not None:
        colorbar = fig.colorbar(last_points, ax=list(axes), fraction=0.028, pad=0.02)
        colorbar.set_label(f"{marker_gene} expression", fontsize=7)
        colorbar.ax.tick_params(labelsize=6)
    fig.subplots_adjust(left=0.07, right=0.91, bottom=0.10, top=0.90, wspace=0.18)
    save_publication_figure(fig, output_path, dpi)


def compute_selected_logfc_matrix(
    adata: ad.AnnData,
    selected_celltypes: Sequence[str],
    selected_genes: Sequence[str],
    reference_condition: str,
) -> pd.DataFrame:
    """Compute Scanpy-style approximate log2 fold changes for selected genes only."""
    available_genes = [gene for gene in selected_genes if gene in adata.var_names]
    work = adata[
        clean_obs_values(adata.obs["condition"]) == str(reference_condition),
        available_genes,
    ]
    celltypes = clean_obs_values(work.obs["celltype"]).to_numpy()
    X = work.X
    rows: List[np.ndarray] = []
    for celltype in selected_celltypes:
        group_mask = celltypes == str(celltype)
        rest_mask = ~group_mask
        if not group_mask.any() or not rest_mask.any():
            rows.append(np.full(len(available_genes), np.nan))
            continue
        group_mean = np.asarray(X[group_mask].mean(axis=0)).ravel()
        rest_mean = np.asarray(X[rest_mask].mean(axis=0)).ravel()
        group_linear = np.maximum(np.expm1(group_mean), 0)
        rest_linear = np.maximum(np.expm1(rest_mean), 0)
        rows.append(np.log2((group_linear + 1e-9) / (rest_linear + 1e-9)))
    frame = pd.DataFrame(rows, index=list(selected_celltypes), columns=available_genes)
    for missing_gene in selected_genes:
        if missing_gene not in frame:
            frame[missing_gene] = np.nan
    return frame[list(selected_genes)]


def logfc_to_rank_matrix(logfc: pd.DataFrame) -> pd.DataFrame:
    ranks = []
    for _, row in logfc.iterrows():
        values = row.to_numpy(dtype=float)
        finite = np.isfinite(values)
        ranked = np.ones(len(values), dtype=float)
        if finite.any():
            fill = np.nanmin(values[finite]) - 1
            ranked = rankdata(np.where(finite, values, fill), method="average")
        ranks.append(ranked)
    return pd.DataFrame(ranks, index=logfc.index, columns=logfc.columns)


def plot_metaq_rank_heatmap(
    rank_matrix: pd.DataFrame,
    title: str,
    output_path: Path,
    dpi: int,
) -> None:
    if rank_matrix.empty:
        return
    fig_width = max(9.0, min(11.5, 0.20 * rank_matrix.shape[1]))
    fig, ax = plt.subplots(figsize=(fig_width, 4.0))
    sns.heatmap(
        rank_matrix,
        cmap=sns.color_palette("Oranges", as_cmap=True),
        vmin=1,
        vmax=rank_matrix.shape[1],
        cbar_kws={"label": "Rank", "shrink": 0.80},
        ax=ax,
    )
    ax.set_title(title, fontsize=DE_TITLE_FONTSIZE)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", labelrotation=60, labelsize=14)
    ax.tick_params(axis="y", labelrotation=0, labelsize=DE_TICK_FONTSIZE)
    colorbar = ax.collections[0].colorbar
    colorbar.ax.tick_params(labelsize=15)
    colorbar.ax.yaxis.label.set_size(17)
    fig.tight_layout()
    save_publication_figure(fig, output_path, dpi)


def plot_metaq_rank_heatmap_pair(
    original_rank: pd.DataFrame,
    method_rank: pd.DataFrame,
    method: str,
    output_path: Path,
    dpi: int,
    show_colorbar: bool = True,
) -> None:
    # Keep the source canvas close to a manuscript page width.  An excessively
    # wide source image is heavily down-scaled by LaTeX and makes nominally
    # reasonable font sizes unreadable in the final composite figure.
    fig_width = max(10.0, min(10.8, 0.18 * original_rank.shape[1]))
    fig, axes = plt.subplots(2, 1, figsize=(fig_width, 8.8), sharex=True)
    fig.subplots_adjust(
        left=0.17,
        right=0.88 if show_colorbar else 0.985,
        bottom=0.36,
        top=0.93,
        hspace=0.46,
    )
    # Both matrices are ranks over the same genes and therefore use exactly the
    # same scale.  One tall shared colorbar is clearer than two redundant bars
    # and, unlike the previous bottom-only placement, visibly applies to both.
    cbar_ax = (
        fig.add_axes([0.915, 0.34, 0.022, 0.36]) if show_colorbar else None
    )
    vmax = original_rank.shape[1]
    for panel_index, (ax, matrix, title) in enumerate([
        (axes[0], original_rank, "Full data differential expression w.r.t. cell types"),
        (axes[1], method_rank, f"{method} differential expression w.r.t. cell types"),
    ]):
        sns.heatmap(
            matrix,
            cmap=sns.color_palette("Oranges", as_cmap=True),
            vmin=1,
            vmax=vmax,
            cbar=show_colorbar and panel_index == 1,
            cbar_ax=cbar_ax if show_colorbar and panel_index == 1 else None,
            cbar_kws={"label": "Rank"} if show_colorbar else None,
            ax=ax,
        )
        ax.set_title(title, fontsize=DE_TITLE_FONTSIZE + 3, pad=11)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(
            axis="y", labelrotation=0, labelsize=DE_TICK_FONTSIZE, pad=5
        )
    if cbar_ax is not None:
        cbar_ax.set_yticks([1, vmax])
        cbar_ax.set_yticklabels(["Low", "High"], fontsize=DE_TICK_FONTSIZE)
        cbar_ax.set_ylabel("Rank", fontsize=DE_AXIS_LABEL_FONTSIZE, labelpad=9)
        cbar_ax.tick_params(length=0, pad=4)
    axes[1].tick_params(axis="x", labelrotation=72, labelsize=15, pad=4)
    for label in axes[1].get_xticklabels():
        label.set_horizontalalignment("right")
        label.set_rotation_mode("anchor")
    save_publication_figure(fig, output_path, dpi)


def save_de_rank_colorbar(
    output_path: Path,
    dpi: int,
    maximum_rank: int,
) -> None:
    """Save the DE-rank scale as a compact standalone horizontal legend."""
    fig = plt.figure(figsize=(6.4, 1.05))
    cbar_ax = fig.add_axes([0.035, 0.28, 0.93, 0.23])
    mappable = plt.cm.ScalarMappable(
        norm=plt.Normalize(vmin=1, vmax=maximum_rank),
        cmap=sns.color_palette("Oranges", as_cmap=True),
    )
    colorbar = fig.colorbar(
        mappable,
        cax=cbar_ax,
        orientation="horizontal",
        ticks=[1, maximum_rank],
    )
    colorbar.outline.set_visible(False)
    colorbar.ax.set_xticklabels(["Low", "High"])
    colorbar.ax.tick_params(
        labelsize=DE_TICK_FONTSIZE,
        length=0,
        pad=2,
    )
    colorbar.ax.set_title(
        "Rank",
        fontsize=DE_AXIS_LABEL_FONTSIZE,
        pad=-5,
    )
    save_publication_figure(fig, output_path, dpi)


def plot_metaq_rank_heatmap_grid(
    rank_matrices: Mapping[str, pd.DataFrame],
    method_order: Sequence[str],
    output_path: Path,
    dpi: int,
) -> None:
    """Show original cells and every metacell method with a shared rank scale."""
    methods = [method for method in method_order if method in rank_matrices]
    if not methods:
        return
    n_cols = 2
    n_rows = int(np.ceil(len(methods) / n_cols))
    first = rank_matrices[methods[0]]
    fig_width = max(13.0, min(14.5, 0.23 * first.shape[1]))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(fig_width, max(9.0, 2.55 * n_rows)),
        squeeze=False,
    )
    vmax = first.shape[1]
    cbar_ax = fig.add_axes([0.925, 0.16, 0.016, 0.22])
    for panel_idx, method in enumerate(methods):
        row_idx, col_idx = divmod(panel_idx, n_cols)
        ax = axes[row_idx, col_idx]
        sns.heatmap(
            rank_matrices[method],
            cmap=sns.color_palette("Oranges", as_cmap=True),
            vmin=1,
            vmax=vmax,
            cbar=panel_idx == len(methods) - 1,
            cbar_ax=cbar_ax if panel_idx == len(methods) - 1 else None,
            cbar_kws={"label": "Rank"},
            ax=ax,
        )
        title = publication_method_label(method)
        ax.set_title(title, fontsize=20)
        ax.set_xlabel("")
        ax.set_ylabel("")
        if col_idx == 0:
            ax.tick_params(axis="y", labelrotation=0, labelsize=16)
        else:
            # Both columns use the same cell-type row order, so repeating the
            # labels wastes space and can overlap the neighboring heatmap.
            ax.set_yticklabels([])
            ax.tick_params(axis="y", left=False)
        if row_idx == n_rows - 1:
            ax.tick_params(axis="x", labelrotation=60, labelsize=14)
            labels = ax.get_xticklabels()
            for label_index, label in enumerate(labels):
                if label_index % 2:
                    label.set_visible(False)
        else:
            ax.set_xticklabels([])
    for panel_idx in range(len(methods), n_rows * n_cols):
        axes.flat[panel_idx].axis("off")
    fig.suptitle(
        "Differential-expression rank preservation across all methods",
        fontsize=23,
        y=0.995,
    )
    fig.subplots_adjust(
        left=0.10,
        right=0.90,
        bottom=0.10,
        top=0.95,
        wspace=0.20,
        hspace=0.55,
    )
    cbar_ax.tick_params(labelsize=17)
    cbar_ax.yaxis.label.set_size(19)
    save_publication_figure(fig, output_path, dpi)


def plot_metaq_rank_consistency(
    concordance: pd.DataFrame,
    selected_celltypes: Sequence[str],
    focal_method: str,
    output_path: Path,
    stats_path: Path,
    dpi: int,
    method_order: Optional[Sequence[str]] = None,
) -> None:
    order = (
        list(method_order)
        if method_order is not None
        else [focal_method]
        + [method for method in METAQ_PAPER_BASELINES if method != focal_method]
    )
    data = concordance[
        concordance["method"].isin(order)
        & concordance["celltype"].astype(str).isin([str(value) for value in selected_celltypes])
    ].copy()
    order = [method for method in order if method in set(data["method"])]
    if focal_method not in order:
        return
    palette = paper_comparison_palette(order, focal_method)
    baseline_comparators = [
        method
        for method in METAQ_PAPER_BASELINES
        if method in order and method != focal_method
    ]
    tests = camp_vs_baseline_ttests(data, "kendall_tau", order)
    save_csv(tests, stats_path)
    annotated_comparator = "SuperCell" if "SuperCell" in order else baseline_comparators[-1]
    annotation_tests = tests[
        (tests["focal_method"] == focal_method)
        & (tests["comparator"] == annotated_comparator)
    ]
    fig_width = max(4.4, 0.78 * len(order) + 1.8)
    fig, ax = plt.subplots(figsize=(fig_width, 4.5))
    sns.boxplot(
        data=data,
        x="method",
        y="kendall_tau",
        order=order,
        hue="method",
        palette=palette,
        legend=False,
        saturation=0.70,
        width=0.72,
        linewidth=0.8,
        fliersize=1.5,
        ax=ax,
    )
    sns.stripplot(
        data=data,
        x="method",
        y="kendall_tau",
        order=order,
        color="black",
        size=2.0,
        alpha=0.75,
        jitter=0.10,
        ax=ax,
    )
    ax.set_title("Rank Consistency", fontsize=10)
    ax.set_xlabel("")
    ax.set_ylabel("Kendall's tau")
    means = data.groupby("method")["kendall_tau"].mean()
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels(
        [f"{method}\nmean={means.get(method, np.nan):.3f}" for method in order],
        rotation=25,
        ha="right",
        fontsize=6.5,
    )
    add_vertical_significance_brackets(ax, order, annotation_tests, focal_method)
    if not annotation_tests.empty:
        n_value = int(annotation_tests.iloc[0]["n_focal"])
        ax.text(
            0.99,
            0.01,
            f"Bracket: {focal_method} vs {annotated_comparator}; "
            f"two-sided independent t-test; n={n_value} cell types/method",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=5.8,
        )
    sns.despine(ax=ax)
    fig.tight_layout()
    save_publication_figure(fig, output_path, dpi)


def select_condition_heatmap_genes(
    condition_de: pd.DataFrame,
    focus_celltype: str,
    methods: Sequence[str],
    max_genes: int = 30,
) -> List[str]:
    original = condition_de[
        (condition_de["method"] == "Original cells")
        & (condition_de["celltype"].astype(str) == focus_celltype)
    ].copy()
    selected: List[str] = []
    original_gene_scores = (
        original.assign(abs_lfc=original["logfoldchanges"].abs())
        .groupby("names", as_index=False)["abs_lfc"]
        .max()
        .nlargest(12, "abs_lfc")
    )
    for gene in original_gene_scores["names"].astype(str):
        if gene not in selected:
            selected.append(gene)
    original_values = original[
        ["contrast", "names", "logfoldchanges"]
    ].rename(columns={"logfoldchanges": "original_lfc"})
    for method in methods:
        method_frame = condition_de[
            (condition_de["method"] == method)
            & (condition_de["celltype"].astype(str) == focus_celltype)
        ].copy()
        method_frame = method_frame.merge(
            original_values,
            on=["contrast", "names"],
            how="inner",
        )
        method_frame["absolute_difference"] = (
            method_frame["logfoldchanges"] - method_frame["original_lfc"]
        ).abs()
        difference_scores = (
            method_frame.groupby("names", as_index=False)["absolute_difference"]
            .max()
            .nlargest(5, "absolute_difference")
        )
        for gene in difference_scores["names"].astype(str):
            if gene not in selected:
                selected.append(gene)
            if len(selected) >= max_genes:
                return selected
    return selected[:max_genes]


def plot_metaq_condition_heatmaps(
    condition_de: pd.DataFrame,
    focus_celltype: str,
    focal_method: str,
    output_dir: Path,
    intermediate_dir: Path,
    dpi: int,
    condition_display_name: str,
) -> None:
    methods = [
        method
        for method in ["Original cells", focal_method, "MetaQ", "SEACells", "SuperCell"]
        if method in set(condition_de["method"])
    ]
    genes = select_condition_heatmap_genes(condition_de, focus_celltype, methods[1:])
    contrasts = sorted(
        condition_de.loc[
            (condition_de["method"] == "Original cells")
            & (condition_de["celltype"].astype(str) == focus_celltype),
            "contrast",
        ].astype(str).unique()
    )
    rows = []
    for method in methods:
        frame = condition_de[
            (condition_de["method"] == method)
            & (condition_de["celltype"].astype(str) == focus_celltype)
        ]
        reference = (
            str(frame["reference"].iloc[0]) if not frame.empty else "reference"
        )
        value_map = {
            (str(row.contrast), str(row.names)): row.logfoldchanges
            for row in frame[["contrast", "names", "logfoldchanges"]].itertuples(index=False)
        }
        for contrast in contrasts:
            for gene in genes:
                rows.append(
                    {
                        "method": method,
                        "celltype": focus_celltype,
                        "contrast": contrast,
                        "reference": reference,
                        "gene": gene,
                        "logfoldchanges": value_map.get((contrast, gene), np.nan),
                    }
                )
    long = pd.DataFrame(rows)
    save_csv(long, intermediate_dir / f"fig5d_{slugify(focal_method).lower()}_values.csv.gz")
    matrix = long.pivot_table(
        index=["method", "contrast"],
        columns="gene",
        values="logfoldchanges",
        aggfunc="mean",
    ).reindex(columns=genes)
    finite_matrix = matrix.to_numpy(dtype=float)
    vmax = max(1.0, float(np.nanquantile(np.abs(finite_matrix), 0.98)))
    for method in methods:
        method_matrix = matrix.xs(method, level="method").reindex(contrasts)
        fig, ax = plt.subplots(
            figsize=(max(7.5, 0.25 * len(genes)), max(2.0, 0.26 * len(contrasts) + 1.3))
        )
        sns.heatmap(
            method_matrix,
            cmap=sns.color_palette("vlag", as_cmap=True),
            vmin=-vmax,
            vmax=vmax,
            center=0,
            cbar_kws={"label": "Value", "shrink": 0.65},
            ax=ax,
        )
        ax.set_title(
            f"{method}: {condition_display_name}-associated differential expression",
            fontsize=9,
        )
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=45, labelsize=6)
        ax.tick_params(axis="y", rotation=0, labelsize=7)
        fig.tight_layout()
        save_publication_figure(
            fig,
            output_dir
            / f"fig5d_{slugify(focal_method).lower()}__{slugify(method).lower()}.png",
            dpi,
        )
    combined = matrix.reindex(
        pd.MultiIndex.from_product([methods, contrasts], names=["method", "contrast"])
    )
    combined.index = [f"{method} | {contrast}" for method, contrast in combined.index]
    fig, ax = plt.subplots(
        figsize=(
            max(7.5, 0.25 * len(genes)),
            max(4.0, 0.20 * len(combined) + 1.8),
        )
    )
    sns.heatmap(
        combined,
        cmap=sns.color_palette("vlag", as_cmap=True),
        vmin=-vmax,
        vmax=vmax,
        center=0,
        cbar_kws={"label": "Value", "shrink": 0.75},
        ax=ax,
    )
    ax.set_title(
        f"{condition_display_name}-associated DE on {focus_celltype} cells",
        fontsize=10,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=45, labelsize=6)
    ax.tick_params(axis="y", rotation=0, labelsize=7)
    fig.tight_layout()
    save_publication_figure(
        fig,
        output_dir / f"fig5d_{slugify(focal_method).lower()}__combined.png",
        dpi,
    )

    # Paper-layout analog: original data, focal method, SEACells and SuperCell
    # in the same 2-by-2 arrangement as MetaQ Fig. 5d. Each row is one
    # condition-versus-reference contrast (one organ contrast for HFA).
    grid_methods = [
        method
        for method in ["Original cells", focal_method, "SEACells", "SuperCell"]
        if method in matrix.index
    ]
    if len(grid_methods) == 4:
        fig, axes = plt.subplots(
            2,
            2,
            figsize=(max(10.0, 0.31 * len(genes)), max(5.0, 0.35 * len(contrasts) + 3.0)),
        )
        for ax, method in zip(axes.flat, grid_methods):
            method_matrix = matrix.xs(method, level="method").reindex(contrasts)
            sns.heatmap(
                method_matrix,
                cmap=sns.color_palette("vlag", as_cmap=True),
                vmin=-vmax,
                vmax=vmax,
                center=0,
                cbar=method in {grid_methods[1], grid_methods[3]},
                cbar_kws={"label": "Value", "shrink": 0.68},
                ax=ax,
            )
            title_method = "Full data" if method == "Original cells" else method
            ax.set_title(
                f"{title_method} differential expression w.r.t. condition",
                fontsize=8.5,
            )
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.tick_params(axis="x", rotation=45, labelsize=5.5)
            ax.tick_params(axis="y", rotation=0, labelsize=6.5)
        reference = str(long["reference"].iloc[0]) if not long.empty else "reference"
        fig.suptitle(
            f"{focus_celltype}: {condition_display_name} contrasts vs {reference}",
            fontsize=10,
            y=1.01,
        )
        fig.tight_layout()
        save_publication_figure(
            fig,
            output_dir / f"fig5d_paper_grid__{slugify(focal_method).lower()}.png",
            dpi,
        )


def plot_metaq_condition_heatmap_all_methods(
    condition_de: pd.DataFrame,
    focus_celltype: str,
    method_order: Sequence[str],
    output_path: Path,
    values_path: Path,
    dpi: int,
    condition_display_name: str,
) -> None:
    """MetaQ Fig. 5d-style condition-DE heatmap containing every method."""
    available = set(condition_de["method"].astype(str))
    methods = ["Original cells"] + [
        method for method in method_order if method in available
    ]
    genes = select_condition_heatmap_genes(condition_de, focus_celltype, methods[1:])
    contrasts = sorted(
        condition_de.loc[
            (condition_de["method"] == "Original cells")
            & (condition_de["celltype"].astype(str) == focus_celltype),
            "contrast",
        ].astype(str).unique()
    )
    rows = []
    for method in methods:
        frame = condition_de[
            (condition_de["method"] == method)
            & (condition_de["celltype"].astype(str) == focus_celltype)
        ]
        reference = (
            str(frame["reference"].iloc[0]) if not frame.empty else "reference"
        )
        value_map = {
            (str(row.contrast), str(row.names)): row.logfoldchanges
            for row in frame[["contrast", "names", "logfoldchanges"]].itertuples(index=False)
        }
        for contrast in contrasts:
            for gene in genes:
                rows.append(
                    {
                        "method": method,
                        "celltype": focus_celltype,
                        "contrast": contrast,
                        "reference": reference,
                        "gene": gene,
                        "logfoldchanges": value_map.get((contrast, gene), np.nan),
                    }
                )
    long = pd.DataFrame(rows)
    save_csv(long, values_path)
    matrix = long.pivot_table(
        index=["method", "contrast"],
        columns="gene",
        values="logfoldchanges",
        aggfunc="mean",
    ).reindex(
        index=pd.MultiIndex.from_product([methods, contrasts], names=["method", "contrast"]),
        columns=genes,
    )
    matrix.index = [f"{method} | {contrast}" for method, contrast in matrix.index]
    vmax = max(1.0, float(np.nanquantile(np.abs(matrix.to_numpy()), 0.98)))
    fig_width = max(10.0, 0.32 * len(genes))
    fig_height = max(5.0, 0.19 * len(matrix) + 1.8)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    sns.heatmap(
        matrix,
        cmap=sns.color_palette("vlag", as_cmap=True),
        vmin=-vmax,
        vmax=vmax,
        center=0,
        cbar_kws={"label": "log fold change", "shrink": 0.70},
        linewidths=0.15,
        linecolor="white",
        ax=ax,
    )
    ax.set_title(
        f"{focus_celltype}: {condition_display_name}-associated differential expression",
        fontsize=10,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=45, labelsize=6)
    ax.tick_params(axis="y", rotation=0, labelsize=7)
    fig.tight_layout()
    save_publication_figure(fig, output_path, dpi)


def plot_metaq_value_consistency(
    correlations: pd.DataFrame,
    focal_method: str,
    output_path: Path,
    stats_path: Path,
    dpi: int,
    method_order: Optional[Sequence[str]] = None,
    condition_display_name: str = "condition",
) -> None:
    order = (
        list(method_order)
        if method_order is not None
        else [focal_method]
        + [method for method in METAQ_PAPER_BASELINES if method != focal_method]
    )
    data = correlations[correlations["method"].isin(order)].copy()
    order = [method for method in order if method in set(data["method"])]
    if focal_method not in order:
        return
    baseline_comparators = [
        method
        for method in METAQ_PAPER_BASELINES
        if method in order and method != focal_method
    ]
    tests = camp_vs_baseline_ttests(data, "pearson_r", order)
    save_csv(tests, stats_path)
    annotated_comparator = "SuperCell" if "SuperCell" in order else baseline_comparators[-1]
    annotation_tests = tests[
        (tests["focal_method"] == focal_method)
        & (tests["comparator"] == annotated_comparator)
    ]
    palette = paper_comparison_palette(order, focal_method)
    fig_width = max(5.0, 6.3 if len(order) > 6 else 5.0)
    fig_height = max(3.7, 0.48 * len(order) + 1.8)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    sns.boxplot(
        data=data,
        x="pearson_r",
        y="method",
        order=order,
        hue="method",
        palette=palette,
        legend=False,
        saturation=0.70,
        width=0.65,
        linewidth=0.8,
        fliersize=1.5,
        ax=ax,
    )
    sns.stripplot(
        data=data,
        x="pearson_r",
        y="method",
        order=order,
        color="black",
        size=1.8,
        alpha=0.55,
        jitter=0.12,
        ax=ax,
    )
    finite = data["pearson_r"].replace([np.inf, -np.inf], np.nan).dropna()
    if not finite.empty:
        lower = max(-1.0, float(np.floor((finite.min() - 0.005) * 100) / 100))
        ax.set_xlim(lower, 1.005)
    ax.set_title("Value Consistency Across Cell Types", fontsize=10)
    ax.set_xlabel(f"Pearson correlation ({condition_display_name}-associated DE)")
    ax.set_ylabel("")
    means = data.groupby("method")["pearson_r"].mean()
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels(
        [f"{method}  (mean={means.get(method, np.nan):.3f})" for method in order],
        fontsize=6.8,
    )
    add_horizontal_significance_brackets(ax, order, annotation_tests, focal_method)
    if not annotation_tests.empty:
        n_value = int(annotation_tests.iloc[0]["n_focal"])
        ax.text(
            0.99,
            0.01,
            f"Bracket: {focal_method} vs {annotated_comparator}; "
            f"two-sided independent t-test; n={n_value} cell types/method",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=5.8,
        )
    sns.despine(ax=ax)
    fig.tight_layout()
    save_publication_figure(fig, output_path, dpi)


def plot_metaq_paper_style_outputs(
    output_root: Path,
    dataset_name: str,
    condition_display_name: str,
    fig4_intermediate_root: Path,
    fig5_intermediate_root: Path,
    primary_target: int,
    camp_variants: Sequence[str],
    marker_gene: str,
    marker_celltype: str,
    focus_celltype: str,
    top_celltypes: int,
    top_genes_per_celltype: int,
    dpi: int,
    fig4_source_override: Optional[Path] = None,
    fig5_source_override: Optional[Path] = None,
    output_subdir: str = "metaq_paper_style",
) -> None:
    """Create one MetaQ-paper-style PNG/PDF per adapted Figure 4/5 panel."""
    requested = int(primary_target)
    fig4_source = (
        Path(fig4_source_override)
        if fig4_source_override is not None
        else fig4_intermediate_root / f"requested_m{requested}"
    )
    fig5_source = (
        Path(fig5_source_override)
        if fig5_source_override is not None
        else fig5_intermediate_root / f"requested_m{requested}"
    )
    if not fig4_source.is_dir() or not fig5_source.is_dir():
        raise FileNotFoundError(
            "MetaQ-style plotting requires completed Fig. 4 and Fig. 5 checkpoints "
            f"for requested m={requested}; Fig. 4 source={fig4_source}; "
            f"Fig. 5 source={fig5_source}"
        )
    figure_root = output_root / "figures" / output_subdir
    fig4_dir = figure_root / "fig4" / "individual_panels"
    fig5_dir = figure_root / "fig5" / "individual_panels"
    intermediate_root = output_root / "intermediate_csv" / output_subdir
    fig4_meta = intermediate_root / "fig4"
    fig5_meta = intermediate_root / "fig5"
    for directory in [fig4_dir, fig5_dir, fig4_meta, fig5_meta]:
        safe_mkdir(directory)

    original_umap = pd.read_csv(fig4_source / "original_unintegrated_umap.csv.gz")
    harmony_umap = pd.read_csv(fig4_source / "original_recovered_umap.csv.gz")
    celltype_palette = category_palette(original_umap["celltype"], "husl")
    batch_palette = category_palette(original_umap["batch"], "hls")
    donor_palette = category_palette(original_umap["donor"], "hls")
    condition_palette = category_palette(original_umap["condition"], "Set2")

    # Fig. 4a and 4b, plus the three Fig. 5a source-data views.
    for color_key, palette in [("celltype", celltype_palette), ("batch", batch_palette)]:
        plot_metaq_style_umap(
            original_umap,
            color_key,
            "Cell Type" if color_key == "celltype" else "Batch",
            fig4_dir / f"fig4a_original__{color_key}.png",
            1.4,
            dpi,
            palette,
        )
        plot_metaq_style_umap(
            harmony_umap,
            color_key,
            "Cell Type" if color_key == "celltype" else "Batch",
            fig4_dir / f"fig4b_harmony__{color_key}.png",
            1.4,
            dpi,
            palette,
        )
    for color_key, palette in [
        ("celltype", celltype_palette),
        ("donor", donor_palette),
        ("condition", condition_palette),
    ]:
        plot_metaq_style_umap(
            original_umap,
            color_key,
            {"celltype": "Cell Type", "donor": "Donor", "condition": "Condition"}[color_key],
            fig5_dir / f"fig5a_original__{color_key}.png",
            1.4,
            dpi,
            palette,
        )

    available_methods: List[str] = []
    for path in sorted(fig4_source.glob("*_metacell_umap.csv.gz")):
        method = str(pd.read_csv(path, nrows=1)["method"].iloc[0])
        if method not in available_methods:
            available_methods.append(method)
        frame = pd.read_csv(path)
        for color_key, palette in [("celltype", celltype_palette), ("batch", batch_palette)]:
            plot_metaq_style_umap(
                frame,
                color_key,
                "Cell Type" if color_key == "celltype" else "Batch",
                fig4_dir
                / f"fig4c_{slugify(method).lower()}_metacells__{color_key}.png",
                10.0,
                dpi,
                palette,
            )
    for path in sorted(fig4_source.glob("*_recovered_umap.csv.gz")):
        frame = pd.read_csv(path)
        method = str(frame["method"].iloc[0])
        if method == "Original cells":
            continue
        for color_key, palette in [("celltype", celltype_palette), ("batch", batch_palette)]:
            plot_metaq_style_umap(
                frame,
                color_key,
                "Cell Type" if color_key == "celltype" else "Batch",
                fig4_dir
                / f"fig4d_{slugify(method).lower()}_recovered__{color_key}.png",
                1.4,
                dpi,
                palette,
            )

    available_set = set(available_methods)
    available_methods = [
        method for method in ALL_METHOD_ORDER if method in available_set
    ] + sorted(available_set.difference(ALL_METHOD_ORDER))

    # Fig. 4e: one Sankey/alluvial panel for Harmony and every recovered method.
    cell_labels = original_umap[["item_id", "celltype"]].rename(columns={"item_id": "cell_id"})
    sankey_rows = []
    sankey_specs = [("Original cells", "integrated_cells", "Harmony")]
    sankey_specs.extend((method, "recovered_cells", method) for method in available_methods)
    for method, representation, display in sankey_specs:
        assignment_stem = "original" if method == "Original cells" else slugify(method).lower()
        assignment_path = fig4_source / f"{assignment_stem}_cluster_assignments.csv.gz"
        if not assignment_path.is_file():
            continue
        assignments = pd.read_csv(assignment_path)
        subset = assignments[
            (assignments["representation"] == representation)
            & np.isclose(assignments["resolution"].astype(float), 1.0)
        ][["cell_id", "cluster"]]
        counts = (
            subset.merge(cell_labels, on="cell_id", how="left")
            .dropna(subset=["celltype"])
            .groupby(["celltype", "cluster"], as_index=False)
            .size()
            .rename(columns={"size": "n_cells"})
        )
        counts["method"] = display
        sankey_rows.append(counts)
        plot_metaq_sankey(
            counts,
            display,
            fig4_dir / f"fig4e_sankey__{slugify(display).lower()}.png",
            dpi,
        )
    if sankey_rows:
        save_csv(pd.concat(sankey_rows, ignore_index=True), fig4_meta / "fig4e_sankey_counts.csv")

    # Fig. 4f: marker expression on a selected cell type, with one batch inset.
    cache_path = (
        output_root
        / "cache"
        / f"{slugify(dataset_name).lower()}_log1p_full_genes.h5ad"
    )
    if cache_path.is_file():
        expression_adata = ad.read_h5ad(cache_path)
        if marker_gene not in expression_adata.var_names:
            raise KeyError(f"MetaQ-style marker gene is absent: {marker_gene}")
        common_ids = original_umap["item_id"].astype(str)
        expression_adata = expression_adata[common_ids].copy()
        expression = expression_adata[:, marker_gene].X
        if sparse.issparse(expression):
            expression = expression.toarray()
        expression_map = pd.Series(
            np.asarray(expression).ravel(), index=expression_adata.obs_names.astype(str)
        )
        marker_frames = []
        marker_specs = [("Harmony", harmony_umap)]
        for path in sorted(fig4_source.glob("*_recovered_umap.csv.gz")):
            frame = pd.read_csv(path)
            method = str(frame["method"].iloc[0])
            if method != "Original cells":
                marker_specs.append((method, frame))
        for display, frame in marker_specs:
            marker_frame = frame.copy()
            marker_frame["expression"] = marker_frame["item_id"].astype(str).map(expression_map)
            marker_frame["method"] = display
            marker_frames.append(marker_frame)
        marker_data = pd.concat(marker_frames, ignore_index=True)
        if marker_celltype == "auto":
            harmony_marker = marker_data[marker_data["method"] == "Harmony"]
            marker_means = (
                harmony_marker.groupby("celltype", observed=True)["expression"]
                .agg(["mean", "size"])
                .query("size >= 20")
                .sort_values(["mean", "size"], ascending=False)
            )
            if marker_means.empty:
                raise ValueError(
                    f"Cannot auto-select a cell type for marker {marker_gene}"
                )
            marker_celltype = str(marker_means.index[0])
            logger.info(
                "Auto-selected marker cell type for %s: %s",
                marker_gene,
                marker_celltype,
            )
        if marker_celltype not in set(marker_data["celltype"].astype(str)):
            raise KeyError(f"MetaQ-style marker cell type is absent: {marker_celltype}")
        celltype_data = marker_data[marker_data["celltype"].astype(str) == marker_celltype]
        inset_batch = str(celltype_data["batch"].astype(str).value_counts().index[0])
        vmax = max(1e-6, float(celltype_data["expression"].quantile(0.99)))
        save_csv(
            marker_data[
                marker_data["celltype"].astype(str) == marker_celltype
            ][["item_id", "method", "celltype", "batch", "UMAP1", "UMAP2", "expression"]],
            fig4_meta / "fig4f_marker_expression.csv.gz",
        )
        for display, frame in marker_data.groupby("method", sort=False):
            plot_metaq_marker_panel(
                frame,
                marker_gene,
                marker_celltype,
                inset_batch,
                f"{marker_gene} ({display})",
                fig4_dir / f"fig4f_{slugify(marker_gene).lower()}__{slugify(display).lower()}.png",
                dpi,
                vmax,
            )
        for focal_method in [DISPLAY_NAME_MAP.get(value, value) for value in camp_variants]:
            plot_metaq_marker_pair(
                marker_data,
                focal_method,
                marker_gene,
                marker_celltype,
                inset_batch,
                fig4_dir
                / f"fig4f_{slugify(marker_gene).lower()}_pair__{slugify(focal_method).lower()}.png",
                dpi,
                vmax,
            )

    # Fig. 4g/h: grouped means with the three resolution scores as black dots.
    metrics = read_csvs(fig4_source.glob("*_clustering_metrics.csv"))
    bar_data = []
    camp_display = [DISPLAY_NAME_MAP.get(value, value) for value in camp_variants]
    all_comparison_methods = [
        method for method in ALL_METHOD_ORDER if method in set(metrics["method"])
    ]
    bar_data.append(
        plot_metaq_grouped_clustering_bars(
            metrics,
            all_comparison_methods,
            None,
            "recovered_cells",
            True,
            "Clustering Performance on Recovered Data — All Methods",
            fig4_dir / "fig4g_recovered_clustering__all_methods.png",
            dpi,
        )
    )
    bar_data.append(
        plot_metaq_grouped_clustering_bars(
            metrics,
            all_comparison_methods,
            None,
            "integrated_metacells_mapped_to_cells",
            False,
            "Clustering Performance on Metacell Data — All Methods",
            fig4_dir / "fig4h_metacell_clustering__all_methods.png",
            dpi,
        )
    )
    for focal_method in camp_display:
        comparison = [focal_method] + [
            method for method in METAQ_PAPER_BASELINES if method in set(metrics["method"])
        ]
        bar_data.append(
            plot_metaq_grouped_clustering_bars(
                metrics,
                comparison,
                focal_method,
                "recovered_cells",
                True,
                "Clustering Performance on Recovered Data",
                fig4_dir / f"fig4g_recovered_clustering__{slugify(focal_method).lower()}.png",
                dpi,
            )
        )
        bar_data.append(
            plot_metaq_grouped_clustering_bars(
                metrics,
                comparison,
                focal_method,
                "integrated_metacells_mapped_to_cells",
                False,
                "Clustering Performance on Metacell Data",
                fig4_dir / f"fig4h_metacell_clustering__{slugify(focal_method).lower()}.png",
                dpi,
            )
        )
    bar_data = [frame for frame in bar_data if not frame.empty]
    if bar_data:
        save_csv(pd.concat(bar_data, ignore_index=True), fig4_meta / "fig4gh_bar_plot_data.csv")

    # Figure 5 selection: MetaQ uses six cell types and ten top genes per type.
    protocol = json.loads((fig5_source / "de_protocol.json").read_text())
    reference_condition = str(protocol["reference_condition"])
    selected_celltypes = (
        original_umap[
            original_umap["condition"].astype(str) == reference_condition
        ]["celltype"]
        .astype(str)
        .value_counts()
        .head(int(top_celltypes))
        .index.tolist()
    )
    original_celltype_de = pd.read_csv(fig5_source / "original_celltype_de.csv.gz")
    selected_genes: List[str] = []
    for celltype in selected_celltypes:
        group = original_celltype_de[
            original_celltype_de["group"].astype(str) == celltype
        ]
        for gene in group.head(int(top_genes_per_celltype))["names"].astype(str):
            if gene not in selected_genes:
                selected_genes.append(gene)
    save_csv(
        pd.DataFrame(
            {
                "selection_order": np.arange(1, len(selected_celltypes) + 1),
                "celltype": selected_celltypes,
                "reference_condition": reference_condition,
            }
        ),
        fig5_meta / "fig5_selected_celltypes.csv",
    )
    save_csv(
        pd.DataFrame(
            {"selection_order": np.arange(1, len(selected_genes) + 1), "gene": selected_genes}
        ),
        fig5_meta / "fig5_selected_genes.csv",
    )

    base = ad.read_h5ad(cache_path)
    base = base[original_umap["item_id"].astype(str)].copy()
    normalized_obs = original_umap.set_index("item_id").reindex(base.obs_names)
    base.obs["celltype"] = normalized_obs["celltype"].astype(str).to_numpy()
    base.obs["condition"] = normalized_obs["condition"].astype(str).to_numpy()
    logfc_frames = []
    original_logfc = compute_selected_logfc_matrix(
        base, selected_celltypes, selected_genes, reference_condition
    )
    original_rank = logfc_to_rank_matrix(original_logfc)
    original_long = original_logfc.stack(dropna=False).rename("logfoldchanges").reset_index()
    original_long.columns = ["celltype", "gene", "logfoldchanges"]
    original_long["method"] = "Original cells"
    logfc_frames.append(original_long)
    plot_metaq_rank_heatmap(
        original_rank,
        "Full data differential expression w.r.t. cell types",
        fig5_dir / "fig5b_rank_heatmap__original_cells.png",
        dpi,
    )
    del base

    method_logfc: Dict[str, pd.DataFrame] = {}
    rank_matrices: Dict[str, pd.DataFrame] = {"Original cells": original_rank}
    for method in available_methods:
        slug = slugify(method).lower()
        h5ad_candidates = [
            fig5_source / f"{slug}_matched_metacells.h5ad",
            fig5_source / f"{slug}_matched10x_metacells.h5ad",
            fig5_source / f"{slug}_native_stratified_metacells.h5ad",
        ]
        h5ad_path = next((path for path in h5ad_candidates if path.is_file()), None)
        if h5ad_path is None:
            continue
        method_adata = ad.read_h5ad(h5ad_path)
        matrix = compute_selected_logfc_matrix(
            method_adata, selected_celltypes, selected_genes, reference_condition
        )
        method_logfc[method] = matrix
        long = matrix.stack(dropna=False).rename("logfoldchanges").reset_index()
        long.columns = ["celltype", "gene", "logfoldchanges"]
        long["method"] = method
        logfc_frames.append(long)
        ranks = logfc_to_rank_matrix(matrix)
        rank_matrices[method] = ranks
        plot_metaq_rank_heatmap(
            ranks,
            f"{method} differential expression w.r.t. cell types",
            fig5_dir / f"fig5b_rank_heatmap__{slug}.png",
            dpi,
        )
        plot_metaq_rank_heatmap_pair(
            original_rank,
            ranks,
            method,
            fig5_dir / f"fig5b_rank_heatmap_pair__{slug}.png",
            dpi,
        )
        del method_adata
    plot_metaq_rank_heatmap_grid(
        rank_matrices,
        ["Original cells"] + ALL_METHOD_ORDER,
        fig5_dir / "fig5b_rank_heatmap_grid__all_methods.png",
        dpi,
    )
    if logfc_frames:
        save_csv(pd.concat(logfc_frames, ignore_index=True), fig5_meta / "fig5b_logfoldchanges.csv.gz")

    concordance = read_csvs(fig5_source.glob("*_celltype_rank_concordance.csv"))
    all_de_methods = [
        method for method in ALL_METHOD_ORDER if method in set(concordance["method"])
    ]
    legacy_fig5c_stats = fig5_meta / "fig5c_ttests__all_methods_camp1_vs_baselines.csv"
    legacy_fig5c_stats.unlink(missing_ok=True)
    plot_metaq_rank_consistency(
        concordance,
        selected_celltypes,
        "CAMP1",
        fig5_dir / "fig5c_rank_consistency__all_methods.png",
        fig5_meta / "fig5c_ttests__all_camp_vs_baselines.csv",
        dpi,
        method_order=all_de_methods,
    )
    for focal_method in camp_display:
        plot_metaq_rank_consistency(
            concordance,
            selected_celltypes,
            focal_method,
            fig5_dir / f"fig5c_rank_consistency__{slugify(focal_method).lower()}.png",
            fig5_meta / f"fig5c_ttests__{slugify(focal_method).lower()}.csv",
            dpi,
        )

    condition_de = read_csvs(fig5_source.glob("*_condition_de.csv.gz"))
    available_focus = sorted(set(condition_de["celltype"].astype(str)))
    if focus_celltype == "auto":
        original_focus = condition_de[condition_de["method"] == "Original cells"]
        focus_celltype = str(
            original_focus.groupby("celltype", observed=True)["contrast"]
            .nunique()
            .sort_values(ascending=False)
            .index[0]
        )
        logger.info(
            "Auto-selected Fig. 5 focus cell type with the broadest contrast coverage: %s",
            focus_celltype,
        )
    elif focus_celltype not in available_focus:
        raise KeyError(
            f"MetaQ-style focus cell type is absent: {focus_celltype}. "
            "Use --metaq-focus-celltype auto for deterministic selection."
        )
    all_condition_methods = [
        method for method in ALL_METHOD_ORDER if method in set(condition_de["method"])
    ]
    plot_metaq_condition_heatmap_all_methods(
        condition_de,
        focus_celltype,
        all_condition_methods,
        fig5_dir / "fig5d_condition_heatmap__all_methods.png",
        fig5_meta / "fig5d_all_methods_values.csv.gz",
        dpi,
        condition_display_name,
    )
    for focal_method in camp_display:
        plot_metaq_condition_heatmaps(
            condition_de,
            focus_celltype,
            focal_method,
            fig5_dir,
            fig5_meta,
            dpi,
            condition_display_name,
        )

    correlations = read_csvs(fig5_source.glob("*_condition_correlations.csv"))
    all_correlation_methods = [
        method for method in ALL_METHOD_ORDER if method in set(correlations["method"])
    ]
    legacy_fig5e_stats = fig5_meta / "fig5e_ttests__all_methods_camp1_vs_baselines.csv"
    legacy_fig5e_stats.unlink(missing_ok=True)
    plot_metaq_value_consistency(
        correlations,
        "CAMP1",
        fig5_dir / "fig5e_value_consistency__all_methods.png",
        fig5_meta / "fig5e_ttests__all_camp_vs_baselines.csv",
        dpi,
        method_order=all_correlation_methods,
        condition_display_name=condition_display_name,
    )
    for focal_method in camp_display:
        plot_metaq_value_consistency(
            correlations,
            focal_method,
            fig5_dir / f"fig5e_value_consistency__{slugify(focal_method).lower()}.png",
            fig5_meta / f"fig5e_ttests__{slugify(focal_method).lower()}.csv",
            dpi,
            condition_display_name=condition_display_name,
        )

    save_json(
        {
            "reference_paper": "MetaQ, Nature Communications 2025, Figs. 4-5",
            "dataset_name": dataset_name,
            "condition_display_name": condition_display_name,
            "requested_native_metacell_target": requested,
            "fig4_source": str(fig4_source),
            "fig5_source": str(fig5_source),
            "fig4_protocol": json.loads(
                (fig4_source / "fig4_protocol.json").read_text()
            ).get("protocol"),
            "fig5_protocol": protocol.get("protocol"),
            "fig5_requested_reduction_rate": protocol.get(
                "requested_reduction_rate"
            ),
            "compression_count_controlled": (
                protocol.get("protocol") == MATCHED_DE_PROTOCOL_VERSION
            ),
            "reference_condition": reference_condition,
            "fig4_marker_gene": marker_gene,
            "fig4_marker_celltype": marker_celltype,
            "fig5_focus_celltype": focus_celltype,
            "fig5_selected_celltypes": selected_celltypes,
            "fig5_selected_genes": selected_genes,
            "individual_png_and_pdf_saved": True,
            "all_method_comparison_panels_saved": True,
            "all_method_order": ALL_METHOD_ORDER,
            "all_camp_vs_baseline_test_tables_saved": True,
            "plot_layout_matches_metaq_fig4": True,
            "plot_layout_matches_metaq_fig5a_to_c": True,
            "scientific_data_identical_to_metaq_paper": False,
            "fig5d_to_e_exact_scientific_structure_available": False,
            "fig5d_to_e_limitation": (
                "This dataset is an adaptation rather than the MetaQ perturbation data. "
                "Every observed condition-versus-reference contrast is shown as its own "
                "row; no perturbation replicates are fabricated."
            ),
            "fig5c_bracket": (
                "One prespecified default-CAMP1-versus-SuperCell bracket, matching "
                "the visual annotation structure of MetaQ Fig. 5c; the exact p-value "
                "and significance category are printed on the figure."
            ),
            "significance_test": "two-sided independent T-test, matching the MetaQ caption",
            "statistical_annotation_note": (
                "All-method Fig. 5c/e panels display one uncluttered CAMP1-versus-"
                "SuperCell bracket. CSV files contain CAMP1 comparisons with every "
                "non-CAMP baseline, including exact p-values, sample sizes, means, "
                "and mean differences. ns means p > 0.05 and is not an equivalence claim."
            ),
        },
        intermediate_root / "metaq_paper_style_protocol.json",
    )

    panel_rows = []
    for panel_path in sorted(figure_root.glob("fig*/individual_panels/*.png")):
        panel_name = panel_path.stem
        paper_panel = panel_name.split("_", 1)[0]
        if paper_panel in {"fig5d", "fig5e"}:
            correspondence = (
                "PBMC COVID-vs-Healthy adaptation; MetaQ used 144 perturbations "
                "per applicable cell type"
            )
        elif paper_panel == "fig4f":
            correspondence = "PBMC marker analog: B-cell MS4A1 replaces alpha-cell TM4SF4"
        else:
            correspondence = "Same panel type and axes; PBMC dataset adaptation"
        panel_rows.append(
            {
                "paper_panel": paper_panel,
                "png": str(panel_path.relative_to(output_root)),
                "pdf": str(panel_path.with_suffix(".pdf").relative_to(output_root)),
                "correspondence": correspondence,
            }
        )
    save_csv(pd.DataFrame(panel_rows), intermediate_root / "panel_manifest.csv")


# =========================================================
# Donor-blocked B-lineage heterogeneity analysis
# =========================================================
def read_integrated_embedding(path: Path) -> Tuple[str, pd.Index, np.ndarray]:
    """Read a saved high-dimensional Harmony/recovered embedding checkpoint."""
    frame = pd.read_csv(path)
    if "item_id" not in frame or "method" not in frame:
        raise ValueError(f"Embedding checkpoint has an unexpected schema: {path}")
    pc_columns = sorted(
        [column for column in frame.columns if re.fullmatch(r"PC\d+", str(column))],
        key=lambda column: int(str(column)[2:]),
    )
    if not pc_columns:
        raise ValueError(f"Embedding checkpoint contains no PC columns: {path}")
    method = str(frame["method"].iloc[0])
    if method == "Original cells":
        method = "Harmony"
    ids = pd.Index(frame["item_id"].astype(str), name="item_id")
    embedding = frame[pc_columns].to_numpy(dtype=np.float32, copy=True)
    return method, ids, embedding


def blineage_method_order(methods: Iterable[str]) -> List[str]:
    preferred = ["Harmony", *ALL_METHOD_ORDER]
    method_set = {str(method) for method in methods}
    ordered = [method for method in preferred if method in method_set]
    ordered.extend(sorted(method_set.difference(ordered)))
    return ordered


def blineage_method_color(method: str) -> str:
    return BLINEAGE_METHOD_PALETTE.get(method, CUSTOM_PALETTE.get(method, "#777777"))


def load_native_assignment_for_target(
    output_root: Path,
    target: int,
    method: str,
) -> pd.Series:
    slug = slugify(method).lower()
    candidate_paths = [
        output_root
        / "generated_partitions"
        / "native_resolution_grid"
        / "camp"
        / f"requested_m{target}"
        / f"{slug}_native_assignment.csv.gz",
        output_root
        / "generated_partitions"
        / "native_resolution_grid"
        / "released_baselines"
        / f"requested_m{target}"
        / f"{slug}_native_assignment.csv.gz",
    ]
    assignment_path = next((path for path in candidate_paths if path.is_file()), None)
    if assignment_path is None:
        raise FileNotFoundError(
            f"No native assignment checkpoint found for {method}, requested m={target}"
        )
    frame = pd.read_csv(assignment_path)
    if not {"cell_id", "metacell"}.issubset(frame.columns):
        raise ValueError(f"Unexpected assignment schema: {assignment_path}")
    assignment = pd.Series(
        frame["metacell"].astype(str).to_numpy(),
        index=frame["cell_id"].astype(str),
        name="metacell",
    )
    if assignment.index.has_duplicates:
        raise ValueError(f"Duplicate cell IDs in {assignment_path}")
    return assignment


def load_matched_assignment_for_target(
    output_root: Path,
    target: int,
    method: str,
) -> pd.Series:
    """Load one exact-count Fig. 4 assignment and enforce its count contract."""
    slug = slugify(method).lower()
    assignment_path = (
        output_root
        / "generated_partitions"
        / "matched_count_grid"
        / f"m{target}"
        / f"{slug}_matched_assignment.csv.gz"
    )
    if not assignment_path.is_file():
        raise FileNotFoundError(
            f"No matched-count assignment found for {method}, m={target}: "
            f"{assignment_path}"
        )
    frame = pd.read_csv(assignment_path)
    if not {"cell_id", "metacell"}.issubset(frame.columns):
        raise ValueError(f"Unexpected assignment schema: {assignment_path}")
    assignment = pd.Series(
        frame["metacell"].astype(str).to_numpy(),
        index=frame["cell_id"].astype(str),
        name="metacell",
    )
    if assignment.index.has_duplicates:
        raise ValueError(f"Duplicate cell IDs in {assignment_path}")
    if int(assignment.nunique()) != int(target):
        raise ValueError(
            f"{method}: expected exactly {target} metacells, found "
            f"{assignment.nunique()} in {assignment_path}"
        )
    return assignment


def safe_correlation(
    x: np.ndarray,
    y: np.ndarray,
    kind: str,
) -> float:
    finite = np.isfinite(x) & np.isfinite(y)
    x = np.asarray(x[finite], dtype=float)
    y = np.asarray(y[finite], dtype=float)
    if len(x) < 3 or np.unique(x).size < 2 or np.unique(y).size < 2:
        return np.nan
    if kind == "pearson":
        return float(pearsonr(x, y).statistic)
    if kind == "spearman":
        return float(spearmanr(x, y).statistic)
    raise ValueError(f"Unsupported correlation kind: {kind}")


def donor_embedding_metrics(
    embedding: np.ndarray,
    metadata: pd.DataFrame,
    celltypes: Sequence[str],
    neighbors: int,
    method: str,
    requested_metacells: int,
    realized_metacells: int,
) -> pd.DataFrame:
    """Evaluate subtype geometry without pooling biological donor replicates."""
    rows: List[Dict[str, object]] = []
    labels_order = [str(value) for value in celltypes]
    for donor in sorted(metadata["donor"].astype(str).unique()):
        donor_mask = metadata["donor"].astype(str).to_numpy() == donor
        X_donor = embedding[donor_mask]
        labels = metadata.loc[donor_mask, "celltype"].astype(str).to_numpy()
        counts = pd.Series(labels).value_counts()
        if len(X_donor) < 3 or counts.size < 2 or counts.min() < 2:
            continue
        silhouette = float(silhouette_score(X_donor, labels, metric="cosine"))
        k_use = min(max(1, neighbors), len(X_donor) - 1)
        neighbor_index = NearestNeighbors(
            n_neighbors=k_use + 1,
            metric="cosine",
            algorithm="brute",
        ).fit(X_donor).kneighbors(X_donor, return_distance=False)[:, 1:]
        purity = float(np.mean(labels[neighbor_index] == labels[:, None]))
        for metric, score in [
            ("within_donor_silhouette_cosine", silhouette),
            ("within_donor_neighbor_purity", purity),
        ]:
            rows.append(
                {
                    "requested_metacells": requested_metacells,
                    "realized_metacells": realized_metacells,
                    "method": method,
                    "donor": donor,
                    "n_cells": len(X_donor),
                    "metric": metric,
                    "score": score,
                }
            )

    donors = sorted(metadata["donor"].astype(str).unique())
    for test_donor in donors:
        train_mask = metadata["donor"].astype(str).to_numpy() != test_donor
        test_mask = ~train_mask
        train_labels = metadata.loc[train_mask, "celltype"].astype(str).to_numpy()
        test_labels = metadata.loc[test_mask, "celltype"].astype(str).to_numpy()
        if not set(labels_order).issubset(set(train_labels)) or not set(labels_order).issubset(
            set(test_labels)
        ):
            continue
        k_use = min(max(1, neighbors), int(train_mask.sum()))
        classifier = KNeighborsClassifier(
            n_neighbors=k_use,
            weights="distance",
            metric="cosine",
            algorithm="brute",
        )
        classifier.fit(embedding[train_mask], train_labels)
        prediction = classifier.predict(embedding[test_mask])
        for metric, score in [
            ("leave_one_donor_balanced_accuracy", balanced_accuracy_score(test_labels, prediction)),
            (
                "leave_one_donor_macro_f1",
                f1_score(
                    test_labels,
                    prediction,
                    labels=labels_order,
                    average="macro",
                    zero_division=0,
                ),
            ),
        ]:
            rows.append(
                {
                    "requested_metacells": requested_metacells,
                    "realized_metacells": realized_metacells,
                    "method": method,
                    "donor": test_donor,
                    "n_cells": int(test_mask.sum()),
                    "metric": metric,
                    "score": float(score),
                }
            )
    return pd.DataFrame(rows)


def donor_mixing_within_blineage_states(
    embedding: np.ndarray,
    metadata: pd.DataFrame,
    celltypes: Sequence[str],
    neighbors: int,
    method: str,
    requested_metacells: int,
    realized_metacells: int,
) -> pd.DataFrame:
    """Measure donor mixing within each biological state using normalized Simpson diversity."""
    rows: List[Dict[str, object]] = []
    for celltype in celltypes:
        state_mask = metadata["celltype"].astype(str).to_numpy() == str(celltype)
        X_state = embedding[state_mask]
        donors = metadata.loc[state_mask, "donor"].astype(str).to_numpy()
        donor_levels = np.unique(donors)
        if len(X_state) < 3 or len(donor_levels) < 2:
            continue
        k_use = min(max(1, neighbors), len(X_state) - 1)
        neighbor_index = NearestNeighbors(
            n_neighbors=k_use + 1,
            metric="cosine",
            algorithm="brute",
        ).fit(X_state).kneighbors(X_state, return_distance=False)[:, 1:]
        mixing_values = []
        denominator = 1.0 - 1.0 / len(donor_levels)
        for neighbor_row in neighbor_index:
            counts = pd.Series(donors[neighbor_row]).value_counts().to_numpy(dtype=float)
            probabilities = counts / counts.sum()
            mixing_values.append((1.0 - float(np.square(probabilities).sum())) / denominator)
        rows.append(
            {
                "requested_metacells": requested_metacells,
                "realized_metacells": realized_metacells,
                "method": method,
                "celltype": str(celltype),
                "n_cells": len(X_state),
                "n_donors": len(donor_levels),
                "donor_mixing_simpson_normalized": float(np.mean(mixing_values)),
            }
        )
    return pd.DataFrame(rows)


def blineage_proportion_preservation(
    assignment: pd.Series,
    metadata: pd.DataFrame,
    celltypes: Sequence[str],
    method: str,
    requested_metacells: int,
    realized_metacells: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Reconstruct donor subtype proportions by majority-labeling native metacells."""
    aligned = assignment.reindex(metadata.index)
    if aligned.isna().any():
        raise ValueError(f"{method}: assignment is missing selected B-lineage cells")
    work = metadata[["donor", "celltype"]].copy()
    work["metacell"] = aligned.astype(str)
    summary_rows: List[Dict[str, object]] = []
    state_rows: List[Dict[str, object]] = []
    for donor, donor_frame in work.groupby("donor", sort=True):
        contingency = pd.crosstab(donor_frame["metacell"], donor_frame["celltype"]).reindex(
            columns=list(celltypes), fill_value=0
        )
        sizes = contingency.sum(axis=1)
        majority = contingency.idxmax(axis=1)
        predicted_counts = pd.Series(0.0, index=list(celltypes))
        for metacell_id, label in majority.items():
            predicted_counts.loc[str(label)] += float(sizes.loc[metacell_id])
        actual_counts = donor_frame["celltype"].value_counts().reindex(celltypes, fill_value=0)
        actual = actual_counts / actual_counts.sum()
        predicted = predicted_counts / predicted_counts.sum()
        absolute_error = (actual - predicted).abs()
        weighted_purity = float(contingency.max(axis=1).sum() / contingency.to_numpy().sum())
        summary_rows.append(
            {
                "requested_metacells": requested_metacells,
                "realized_metacells": realized_metacells,
                "method": method,
                "donor": str(donor),
                "n_cells": len(donor_frame),
                "n_metacells_represented": contingency.shape[0],
                "total_variation_error": float(0.5 * absolute_error.sum()),
                "maximum_absolute_proportion_error": float(absolute_error.max()),
                "weighted_within_lineage_metacell_purity": weighted_purity,
            }
        )
        for celltype in celltypes:
            state_rows.append(
                {
                    "requested_metacells": requested_metacells,
                    "realized_metacells": realized_metacells,
                    "method": method,
                    "donor": str(donor),
                    "celltype": str(celltype),
                    "actual_cell_proportion": float(actual.loc[celltype]),
                    "majority_metacell_reconstructed_proportion": float(predicted.loc[celltype]),
                    "absolute_proportion_error": float(absolute_error.loc[celltype]),
                }
            )
    return pd.DataFrame(summary_rows), pd.DataFrame(state_rows)


def marker_reconstruction_metrics(
    assignment: pd.Series,
    expression: pd.DataFrame,
    selected_metadata: pd.DataFrame,
    method: str,
    requested_metacells: int,
    realized_metacells: int,
) -> pd.DataFrame:
    """Compare original marker expression with native-metacell mean reconstruction."""
    aligned = assignment.reindex(expression.index)
    if aligned.isna().any():
        raise ValueError(f"{method}: assignment is missing evaluation cells")
    mean_expression = expression.groupby(aligned.astype(str), sort=False).mean()
    reconstructed = mean_expression.loc[aligned.loc[selected_metadata.index].astype(str)].copy()
    reconstructed.index = selected_metadata.index
    rows: List[Dict[str, object]] = []
    for donor, donor_metadata in selected_metadata.groupby("donor", sort=True):
        ids = donor_metadata.index
        for gene in expression.columns:
            original_values = expression.loc[ids, gene].to_numpy(dtype=float)
            reconstructed_values = reconstructed.loc[ids, gene].to_numpy(dtype=float)
            difference = reconstructed_values - original_values
            rows.append(
                {
                    "requested_metacells": requested_metacells,
                    "realized_metacells": realized_metacells,
                    "method": method,
                    "donor": str(donor),
                    "gene": str(gene),
                    "n_cells": len(ids),
                    "pearson": safe_correlation(original_values, reconstructed_values, "pearson"),
                    "spearman": safe_correlation(original_values, reconstructed_values, "spearman"),
                    "mae": float(np.mean(np.abs(difference))),
                    "rmse": float(np.sqrt(np.mean(np.square(difference)))),
                }
            )
    return pd.DataFrame(rows)


def plot_blineage_annotation_facets(
    frame: pd.DataFrame,
    method: str,
    celltypes: Sequence[str],
    donors: Sequence[str],
    condition: str,
    output_path: Path,
    dpi: int,
) -> None:
    palette = category_palette(list(celltypes), "tab10")
    columns = [(f"All {condition}", frame)] + [
        (str(donor), frame[frame["donor"].astype(str) == str(donor)]) for donor in donors
    ]
    fig, axes = plt.subplots(1, len(columns), figsize=(3.0 * len(columns), 3.0))
    axes = np.atleast_1d(axes)
    for panel_index, (ax, (title, subset)) in enumerate(zip(axes, columns)):
        for celltype in celltypes:
            points = subset[subset["celltype"].astype(str) == str(celltype)]
            ax.scatter(
                points["UMAP1"],
                points["UMAP2"],
                s=3.0,
                linewidths=0,
                color=palette[str(celltype)],
                label=str(celltype),
                rasterized=True,
            )
        ax.set_title(title, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("UMAP1", fontsize=7)
        if panel_index == 0:
            ax.set_ylabel("UMAP2", fontsize=7)
        else:
            ax.set_ylabel("")
    display_method = publication_method_label(method)
    display = (
        "Full-cell Harmony"
        if method == "Harmony"
        else f"Harmony + {display_method}"
    )
    fig.suptitle(f"B-lineage states ({display})", fontsize=11)
    fig.subplots_adjust(left=0.04, right=0.99, bottom=0.12, top=0.84, wspace=0.08)
    save_publication_figure(fig, output_path, dpi)


def save_blineage_state_legend(
    celltypes: Sequence[str],
    output_path: Path,
    dpi: int,
) -> None:
    """Save one frameless horizontal legend shared by all B-lineage UMAPs."""
    palette = category_palette(list(celltypes), "tab10")
    labels = [str(celltype) for celltype in celltypes]
    handles = [
        Line2D(
            [0],
            [0],
            color=palette[label],
            marker="o",
            linestyle="none",
            markersize=4.5,
            label=label,
        )
        for label in labels
    ]
    fig = plt.figure(figsize=(6.2, 0.38))
    legend = fig.legend(
        handles=handles,
        labels=labels,
        loc="center",
        ncol=len(labels),
        frameon=False,
        handlelength=0.8,
        handletextpad=0.28,
        columnspacing=0.75,
        borderaxespad=0,
        fontsize=7.5,
    )
    safe_mkdir(output_path.parent)
    for destination in [output_path, output_path.with_suffix(".pdf")]:
        fig.savefig(
            destination,
            dpi=dpi if destination.suffix == ".png" else None,
            bbox_inches="tight",
            bbox_extra_artists=(legend,),
            pad_inches=0.003,
            facecolor="white",
        )
    plt.close(fig)


def save_horizontal_expression_legend(
    gene: str,
    vmax: float,
    output_path: Path,
    dpi: int,
) -> None:
    """Save a compact horizontal color scale shared by one marker's maps."""
    norm = matplotlib.colors.Normalize(vmin=0.0, vmax=float(vmax))
    scalar = matplotlib.cm.ScalarMappable(norm=norm, cmap="coolwarm")
    scalar.set_array([])
    fig = plt.figure(figsize=(3.8, 0.62))
    colorbar_axis = fig.add_axes([0.035, 0.58, 0.93, 0.20])
    colorbar = fig.colorbar(
        scalar,
        cax=colorbar_axis,
        orientation="horizontal",
    )
    colorbar.outline.set_visible(False)
    colorbar.set_ticks([0.0, float(vmax) / 2.0, float(vmax)])
    colorbar.ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.1f"))
    colorbar.ax.tick_params(axis="x", labelsize=6.5, length=2.0, pad=1.0)
    colorbar.set_label(
        f"{gene} normalized expression",
        fontsize=7.5,
        labelpad=1.5,
    )
    safe_mkdir(output_path.parent)
    for destination in [output_path, output_path.with_suffix(".pdf")]:
        fig.savefig(
            destination,
            dpi=dpi if destination.suffix == ".png" else None,
            bbox_inches="tight",
            pad_inches=0.003,
            facecolor="white",
        )
    plt.close(fig)


def plot_blineage_marker_method_vs_harmony(
    frames: Mapping[str, pd.DataFrame],
    focal_method: str,
    gene: str,
    donors: Sequence[str],
    condition: str,
    output_path: Path,
    dpi: int,
    vmax_override: Optional[float] = None,
) -> None:
    if focal_method not in frames or "Harmony" not in frames:
        return
    methods = [focal_method, "Harmony"]
    vmax = (
        float(vmax_override)
        if vmax_override is not None
        else max(
            1e-6,
            float(
                pd.concat(
                    [frames[method][gene] for method in methods],
                    ignore_index=True,
                ).quantile(0.99)
            ),
        )
    )
    columns = [f"All {condition}", *[str(donor) for donor in donors]]
    fig, axes = plt.subplots(
        2,
        len(columns),
        figsize=(3.0 * len(columns), 5.8),
        squeeze=False,
    )
    last_points = None
    for row_index, method in enumerate(methods):
        method_frame = frames[method]
        for column_index, column in enumerate(columns):
            ax = axes[row_index, column_index]
            subset = (
                method_frame
                if column_index == 0
                else method_frame[method_frame["donor"].astype(str) == column]
            )
            last_points = ax.scatter(
                subset["UMAP1"],
                subset["UMAP2"],
                c=subset[gene],
                cmap="coolwarm",
                vmin=0,
                vmax=vmax,
                s=3.0,
                linewidths=0,
                rasterized=True,
            )
            if row_index == 0:
                ax.set_title(column, fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel("UMAP1", fontsize=7)
            if column_index == 0:
                method_label = (
                    "Full-cell Harmony"
                    if method == "Harmony"
                    else f"Harmony + {publication_method_label(method)}"
                )
                ax.set_ylabel(f"{method_label}\nUMAP2", fontsize=8)
            else:
                ax.set_ylabel("")
    fig.suptitle(f"{gene} across donor-replicated B-lineage states", fontsize=11)
    fig.subplots_adjust(left=0.09, right=0.995, bottom=0.08, top=0.88, wspace=0.08, hspace=0.10)
    save_publication_figure(fig, output_path, dpi)


def plot_resolution_metric(
    frame: pd.DataFrame,
    metric: str,
    ylabel: str,
    output_path: Path,
    dpi: int,
) -> None:
    subset = frame[frame["metric"] == metric].copy()
    if subset.empty:
        return
    methods = blineage_method_order(subset["method"].astype(str).unique())
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for method_index, method in enumerate(methods):
        method_frame = subset[subset["method"].astype(str) == method]
        summary = method_frame.groupby("requested_metacells", as_index=False)["score"].mean()
        ax.plot(
            summary["requested_metacells"],
            summary["score"],
            color=blineage_method_color(method),
            marker="o",
            linewidth=1.5,
            markersize=4,
            label=method,
        )
        jitter = (method_index - (len(methods) - 1) / 2.0) * 3.0
        ax.scatter(
            method_frame["requested_metacells"].astype(float) + jitter,
            method_frame["score"],
            color=blineage_method_color(method),
            s=9,
            alpha=0.38,
            linewidths=0,
        )
    ax.set_xlabel("Requested metacells")
    ax.set_ylabel(ylabel)
    ax.set_xticks(sorted(subset["requested_metacells"].unique()))
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False, fontsize=7)
    sns.despine(ax=ax)
    fig.tight_layout()
    save_publication_figure(fig, output_path, dpi)


def plot_blineage_summary_panels(
    donor_metrics: pd.DataFrame,
    proportion_summary: pd.DataFrame,
    mixing_summary: pd.DataFrame,
    marker_metrics: pd.DataFrame,
    output_dir: Path,
    dpi: int,
) -> None:
    safe_mkdir(output_dir)
    metric_labels = {
        "within_donor_silhouette_cosine": "Within-donor silhouette (cosine)",
        "within_donor_neighbor_purity": "Within-donor neighborhood purity",
        "leave_one_donor_balanced_accuracy": "Leave-one-donor balanced accuracy",
        "leave_one_donor_macro_f1": "Leave-one-donor macro-F1",
    }
    for metric, ylabel in metric_labels.items():
        plot_resolution_metric(
            donor_metrics,
            metric,
            ylabel,
            output_dir / f"blineage_{metric}.png",
            dpi,
        )

    methods = blineage_method_order(proportion_summary["method"].astype(str).unique())
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0))
    for ax, column, ylabel in [
        (axes[0], "total_variation_error", "Subtype proportion error (lower is better)"),
        (
            axes[1],
            "weighted_within_lineage_metacell_purity",
            "Within-lineage metacell purity (higher is better)",
        ),
    ]:
        for method in methods:
            method_frame = proportion_summary[proportion_summary["method"] == method]
            summary = method_frame.groupby("requested_metacells", as_index=False)[column].mean()
            ax.plot(
                summary["requested_metacells"],
                summary[column],
                marker="o",
                linewidth=1.4,
                markersize=4,
                color=blineage_method_color(method),
                label=method,
            )
        ax.set_xlabel("Requested metacells")
        ax.set_ylabel(ylabel)
        ax.set_xticks(sorted(proportion_summary["requested_metacells"].unique()))
        sns.despine(ax=ax)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, bbox_to_anchor=(1.01, 0.88), loc="upper left", frameon=False, fontsize=7)
    fig.tight_layout(rect=[0, 0, 0.88, 1])
    save_publication_figure(fig, output_dir / "blineage_proportion_preservation.png", dpi)

    if not mixing_summary.empty:
        mixing_plot = mixing_summary.groupby(
            ["requested_metacells", "method"], as_index=False
        )["donor_mixing_simpson_normalized"].mean()
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        for method in blineage_method_order(mixing_plot["method"].unique()):
            method_frame = mixing_plot[mixing_plot["method"] == method]
            ax.plot(
                method_frame["requested_metacells"],
                method_frame["donor_mixing_simpson_normalized"],
                marker="o",
                linewidth=1.5,
                markersize=4,
                color=blineage_method_color(method),
                label=method,
            )
        ax.set_xlabel("Requested metacells")
        ax.set_ylabel("Donor mixing within B-lineage states")
        ax.set_xticks(sorted(mixing_plot["requested_metacells"].unique()))
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False, fontsize=7)
        sns.despine(ax=ax)
        fig.tight_layout()
        save_publication_figure(fig, output_dir / "blineage_within_state_donor_mixing.png", dpi)

    if not marker_metrics.empty:
        marker_plot = marker_metrics.groupby(
            ["requested_metacells", "method"], as_index=False
        )["spearman"].mean()
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        for method in blineage_method_order(marker_plot["method"].unique()):
            method_frame = marker_plot[marker_plot["method"] == method]
            ax.plot(
                method_frame["requested_metacells"],
                method_frame["spearman"],
                marker="o",
                linewidth=1.5,
                markersize=4,
                color=blineage_method_color(method),
                label=method,
            )
        ax.set_xlabel("Requested metacells")
        ax.set_ylabel("Marker reconstruction Spearman correlation")
        ax.set_xticks(sorted(marker_plot["requested_metacells"].unique()))
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False, fontsize=7)
        sns.despine(ax=ax)
        fig.tight_layout()
        save_publication_figure(fig, output_dir / "blineage_marker_reconstruction.png", dpi)


def run_blineage_analysis(args: argparse.Namespace, output_root: Path) -> None:
    """Run donor-blocked B-lineage validation at exact shared metacell counts."""
    intermediate_root = (
        output_root / "intermediate_csv" / str(args.blineage_output_subdir)
    )
    figure_root = output_root / "figures" / str(args.blineage_output_subdir)
    quantitative_figure_dir = figure_root / "quantitative"
    primary_figure_dir = figure_root / f"primary_m{args.fig4_primary_target}"
    annotation_dir = primary_figure_dir / "subtype_annotation"
    marker_pair_dir = primary_figure_dir / "marker_method_vs_harmony"
    standalone_legend_dir = primary_figure_dir / "standalone_legends"
    for path in [
        intermediate_root,
        quantitative_figure_dir,
        annotation_dir,
        marker_pair_dir,
        standalone_legend_dir,
    ]:
        safe_mkdir(path)

    fig4_root = (
        output_root / "intermediate_csv" / "fig4_batch_integration" / "matched_counts"
    )
    primary_dir = fig4_root / f"m{args.fig4_primary_target}"
    metadata_path = primary_dir / "original_recovered_umap.csv.gz"
    original_embedding_path = primary_dir / "original_integrated_embedding.csv.gz"
    expression_path = output_root / "cache" / "pbmc_log1p_full_genes.h5ad"
    for required_path in [metadata_path, original_embedding_path, expression_path]:
        if not required_path.is_file():
            raise FileNotFoundError(
                f"B-lineage analysis needs the completed Fig. 4 checkpoint: {required_path}"
            )

    metadata = pd.read_csv(metadata_path).set_index("item_id")
    metadata.index = metadata.index.astype(str)
    for column in ["celltype", "batch", "donor", "condition"]:
        metadata[column] = metadata[column].astype(str)
    requested_celltypes = [str(value) for value in args.blineage_celltypes]
    missing_celltypes = sorted(set(requested_celltypes).difference(metadata["celltype"].unique()))
    if missing_celltypes:
        raise ValueError(f"B-lineage cell types are absent: {missing_celltypes}")

    lineage_metadata = metadata[metadata["celltype"].isin(requested_celltypes)].copy()
    if args.blineage_condition == "auto":
        condition_counts = lineage_metadata["condition"].value_counts()
        non_reference = condition_counts[
            ~condition_counts.index.str.lower().isin({"healthy", "control"})
        ]
        selected_condition = str(
            (non_reference if not non_reference.empty else condition_counts).index[0]
        )
    else:
        selected_condition = str(args.blineage_condition)
    if selected_condition not in set(lineage_metadata["condition"]):
        raise ValueError(
            f"B-lineage condition '{selected_condition}' is absent; found "
            f"{sorted(lineage_metadata['condition'].unique())}"
        )

    condition_metadata = lineage_metadata[
        lineage_metadata["condition"] == selected_condition
    ].copy()
    donor_counts = pd.crosstab(condition_metadata["donor"], condition_metadata["celltype"]).reindex(
        columns=requested_celltypes, fill_value=0
    )
    donor_counts.insert(0, "donor", donor_counts.index.astype(str))
    donor_counts = donor_counts.reset_index(drop=True)
    donor_counts["minimum_cells_across_states"] = donor_counts[requested_celltypes].min(axis=1)
    donor_counts["eligible"] = (
        donor_counts["minimum_cells_across_states"] >= args.blineage_min_cells_per_state
    )
    requested_donors = [str(value) for value in args.blineage_donors]
    if not requested_donors or requested_donors == ["auto"]:
        selected_donors = donor_counts.loc[donor_counts["eligible"], "donor"].tolist()
    else:
        selected_donors = requested_donors
        invalid = sorted(set(selected_donors).difference(donor_counts.loc[donor_counts["eligible"], "donor"]))
        if invalid:
            raise ValueError(
                "Requested B-lineage donors do not meet the per-state minimum "
                f"of {args.blineage_min_cells_per_state}: {invalid}"
            )
    if len(selected_donors) < 2:
        raise ValueError(
            "B-lineage validation requires at least two eligible biological donors; "
            f"found {selected_donors}"
        )
    donor_counts["selected_for_primary_analysis"] = donor_counts["donor"].isin(selected_donors)
    save_csv(donor_counts, intermediate_root / "blineage_donor_state_counts.csv")

    selected_metadata = condition_metadata[
        condition_metadata["donor"].isin(selected_donors)
    ].copy()
    selected_metadata = selected_metadata.loc[
        metadata.index.intersection(selected_metadata.index, sort=False)
    ]
    save_csv(
        selected_metadata.reset_index(),
        intermediate_root / "blineage_selected_cells.csv.gz",
    )

    expression_adata = ad.read_h5ad(expression_path)
    available_markers = [
        str(gene) for gene in args.blineage_marker_genes if str(gene) in expression_adata.var_names
    ]
    missing_markers = [
        str(gene) for gene in args.blineage_marker_genes if str(gene) not in expression_adata.var_names
    ]
    if missing_markers:
        logger.warning("Skipping unavailable B-lineage markers: %s", ", ".join(missing_markers))
    if not available_markers:
        raise ValueError("None of the requested B-lineage markers is present in the cache")
    expression_adata = expression_adata[metadata.index, available_markers].copy()
    expression_values = expression_adata.X
    if sparse.issparse(expression_values):
        expression_values = expression_values.toarray()
    expression = pd.DataFrame(
        np.asarray(expression_values, dtype=np.float32),
        index=metadata.index,
        columns=available_markers,
    )
    del expression_adata

    original_method, original_ids, original_embedding = read_integrated_embedding(
        original_embedding_path
    )
    if original_method != "Harmony":
        raise ValueError(f"Unexpected original embedding method: {original_method}")
    original_lookup = pd.DataFrame(original_embedding, index=original_ids)

    donor_metric_frames: List[pd.DataFrame] = []
    mixing_frames: List[pd.DataFrame] = []
    proportion_frames: List[pd.DataFrame] = []
    proportion_state_frames: List[pd.DataFrame] = []
    marker_metric_frames: List[pd.DataFrame] = []
    realized_by_target_method: Dict[Tuple[int, str], int] = {}
    methods_seen = {"Harmony"}

    for target in args.fig4_target_metacells:
        target_dir = fig4_root / f"m{target}"
        protocol_path = target_dir / "fig4_protocol.json"
        if not protocol_path.is_file():
            raise FileNotFoundError(f"Missing completed Fig. 4 target: {protocol_path}")
        protocol = json.loads(protocol_path.read_text())
        if protocol.get("protocol") != MATCHED_FIG4_PROTOCOL_VERSION:
            raise ValueError(
                f"B-lineage analysis requires {MATCHED_FIG4_PROTOCOL_VERSION}; "
                f"found {protocol.get('protocol')} in {protocol_path}"
            )
        if not bool(protocol.get("exact_count_matched", False)):
            raise ValueError(f"Fig. 4 target is not exact-count matched: {protocol_path}")
        method_realized = {
            str(method): int(value)
            for method, value in protocol.get("method_realized_metacells", {}).items()
        }
        embeddings: Dict[str, Tuple[pd.Index, np.ndarray]] = {
            "Harmony": (original_lookup.index, original_lookup.to_numpy(dtype=np.float32))
        }
        realized_by_target_method[(int(target), "Harmony")] = metadata.shape[0]
        for path in sorted(target_dir.glob("*_recovered_integrated_embedding.csv.gz")):
            method, ids, values = read_integrated_embedding(path)
            embeddings[method] = (ids, values)
            realized_by_target_method[(int(target), method)] = method_realized.get(
                method, -1
            )
            methods_seen.add(method)

        selected_ids = selected_metadata.index
        for method, (ids, values) in embeddings.items():
            lookup = pd.DataFrame(values, index=ids)
            missing_ids = selected_ids.difference(lookup.index)
            if len(missing_ids):
                raise ValueError(
                    f"{method}, requested m={target}: missing {len(missing_ids)} selected cells"
                )
            X_selected = lookup.loc[selected_ids].to_numpy(dtype=np.float32)
            realized = realized_by_target_method[(int(target), method)]
            donor_metric_frames.append(
                donor_embedding_metrics(
                    X_selected,
                    selected_metadata,
                    requested_celltypes,
                    args.blineage_neighbors,
                    method,
                    int(target),
                    realized,
                )
            )
            mixing_frames.append(
                donor_mixing_within_blineage_states(
                    X_selected,
                    selected_metadata,
                    requested_celltypes,
                    args.blineage_neighbors,
                    method,
                    int(target),
                    realized,
                )
            )

        for method in sorted(method_realized):
            assignment = load_matched_assignment_for_target(
                output_root, int(target), method
            )
            realized = method_realized[method]
            if int(realized) != int(target):
                raise ValueError(
                    f"{method}: matched Fig. 4 protocol reports {realized}, "
                    f"expected {target}"
                )
            proportion_summary, proportion_states = blineage_proportion_preservation(
                assignment,
                selected_metadata,
                requested_celltypes,
                method,
                int(target),
                realized,
            )
            proportion_frames.append(proportion_summary)
            proportion_state_frames.append(proportion_states)
            marker_metric_frames.append(
                marker_reconstruction_metrics(
                    assignment,
                    expression,
                    selected_metadata,
                    method,
                    int(target),
                    realized,
                )
            )

        harmony_proportion_rows = []
        harmony_marker_rows = []
        for donor, donor_frame in selected_metadata.groupby("donor", sort=True):
            harmony_proportion_rows.append(
                {
                    "requested_metacells": int(target),
                    "realized_metacells": metadata.shape[0],
                    "method": "Harmony",
                    "donor": str(donor),
                    "n_cells": len(donor_frame),
                    "n_metacells_represented": len(donor_frame),
                    "total_variation_error": 0.0,
                    "maximum_absolute_proportion_error": 0.0,
                    "weighted_within_lineage_metacell_purity": 1.0,
                }
            )
            for gene in available_markers:
                harmony_marker_rows.append(
                    {
                        "requested_metacells": int(target),
                        "realized_metacells": metadata.shape[0],
                        "method": "Harmony",
                        "donor": str(donor),
                        "gene": gene,
                        "n_cells": len(donor_frame),
                        "pearson": 1.0,
                        "spearman": 1.0,
                        "mae": 0.0,
                        "rmse": 0.0,
                    }
                )
        proportion_frames.append(pd.DataFrame(harmony_proportion_rows))
        marker_metric_frames.append(pd.DataFrame(harmony_marker_rows))

    donor_metrics = pd.concat(donor_metric_frames, ignore_index=True)
    mixing_summary = pd.concat(mixing_frames, ignore_index=True)
    proportion_summary = pd.concat(proportion_frames, ignore_index=True)
    proportion_states = pd.concat(proportion_state_frames, ignore_index=True)
    marker_metrics = pd.concat(marker_metric_frames, ignore_index=True)
    save_csv(donor_metrics, intermediate_root / "blineage_donor_embedding_metrics.csv")
    save_csv(mixing_summary, intermediate_root / "blineage_within_state_donor_mixing.csv")
    save_csv(proportion_summary, intermediate_root / "blineage_proportion_preservation.csv")
    save_csv(proportion_states, intermediate_root / "blineage_proportion_state_details.csv")
    save_csv(marker_metrics, intermediate_root / "blineage_marker_reconstruction_metrics.csv")

    primary_frames: Dict[str, pd.DataFrame] = {}
    primary_rows: List[pd.DataFrame] = []
    for path in sorted(primary_dir.glob("*_recovered_umap.csv.gz")):
        frame = pd.read_csv(path)
        method = str(frame["method"].iloc[0])
        if method == "Original cells":
            method = "Harmony"
        frame["item_id"] = frame["item_id"].astype(str)
        frame = frame.set_index("item_id").loc[selected_metadata.index].copy()
        frame["method"] = method
        for gene in available_markers:
            frame[gene] = expression.loc[frame.index, gene]
        primary_frames[method] = frame
        primary_rows.append(frame.reset_index())
    save_csv(
        pd.concat(primary_rows, ignore_index=True),
        intermediate_root / f"blineage_primary_m{args.fig4_primary_target}_marker_coordinates.csv.gz",
    )

    for method in blineage_method_order(primary_frames):
        plot_blineage_annotation_facets(
            primary_frames[method],
            method,
            requested_celltypes,
            selected_donors,
            selected_condition,
            annotation_dir / f"blineage_subtypes__{slugify(method).lower()}.png",
            args.dpi,
        )
    save_blineage_state_legend(
        requested_celltypes,
        standalone_legend_dir / "blineage_state_legend.png",
        args.dpi,
    )
    plot_genes = [gene for gene in args.blineage_plot_genes if gene in available_markers]
    for gene in plot_genes:
        shared_vmax = max(
            1e-6,
            float(
                pd.concat(
                    [frame[gene] for frame in primary_frames.values()],
                    ignore_index=True,
                ).quantile(0.99)
            ),
        )
        for method in blineage_method_order(primary_frames):
            if method == "Harmony":
                continue
            plot_blineage_marker_method_vs_harmony(
                primary_frames,
                method,
                gene,
                selected_donors,
                selected_condition,
                marker_pair_dir
                / f"blineage_{slugify(gene).lower()}__{slugify(method).lower()}_vs_harmony.png",
                args.dpi,
                vmax_override=shared_vmax,
            )
        save_horizontal_expression_legend(
            gene,
            shared_vmax,
            standalone_legend_dir
            / f"blineage_{slugify(gene).lower()}_normalized_expression_legend.png",
            args.dpi,
        )

    plot_blineage_summary_panels(
        donor_metrics,
        proportion_summary,
        mixing_summary,
        marker_metrics,
        quantitative_figure_dir,
        args.dpi,
    )

    method_summary = (
        donor_metrics.groupby(["requested_metacells", "method", "metric"], as_index=False)[
            "score"
        ]
        .mean()
        .rename(columns={"score": "mean_across_selected_donors"})
    )
    save_csv(method_summary, intermediate_root / "blineage_method_metric_means.csv")
    save_json(
        {
            "protocol": BLINEAGE_PROTOCOL_VERSION,
            "source_fig4_protocol": MATCHED_FIG4_PROTOCOL_VERSION,
            "comparison_mode": "exact_global_metacell_count",
            "exact_count_matched": True,
            "requested_metacell_grid": args.fig4_target_metacells,
            "primary_marker_target": args.fig4_primary_target,
            "condition": selected_condition,
            "celltypes": requested_celltypes,
            "selected_donors": selected_donors,
            "minimum_cells_per_state_per_donor": args.blineage_min_cells_per_state,
            "marker_genes_evaluated": available_markers,
            "marker_genes_plotted": plot_genes,
            "methods": blineage_method_order(methods_seen),
            "primary_endpoints": [
                "within-donor subtype silhouette",
                "within-donor neighborhood purity",
                "leave-one-donor balanced accuracy",
                "leave-one-donor macro-F1",
                "donor mixing within each B-lineage state",
                "majority-metacell subtype proportion error",
                "marker-expression reconstruction",
            ],
            "statistical_unit": "biological donor",
            "claim_guardrail": (
                "This analysis tests preservation of known donor-reproducible B-lineage "
                "states. It does not by itself establish a novel B-cell state or prove "
                "batch correction from a marker UMAP."
            ),
            "outputs_are_additive": True,
            "output_subdir": str(args.blineage_output_subdir),
        },
        intermediate_root / "blineage_analysis_protocol.json",
    )
    logger.info(
        "B-lineage analysis completed | condition=%s donors=%s output=%s",
        selected_condition,
        ",".join(selected_donors),
        figure_root,
    )


# =========================================================
# Focused manuscript figure set
# =========================================================
def summarize_mean_sem(
    frame: pd.DataFrame,
    group_columns: Sequence[str],
    value_column: str,
) -> pd.DataFrame:
    summary = (
        frame.groupby(list(group_columns), as_index=False)[value_column]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "mean", "std": "std", "count": "n"})
    )
    summary["sem"] = summary["std"].fillna(0.0) / np.sqrt(
        summary["n"].clip(lower=1)
    )
    return summary


def data_driven_y_limits(
    summary: pd.DataFrame,
    minimum_span: float = 0.0,
    padding_fraction: float = 0.08,
    bounds: Optional[Tuple[Optional[float], Optional[float]]] = None,
    anchor_zero: bool = False,
) -> Optional[Tuple[float, float]]:
    """Choose readable, endpoint-aware limits from means and their SEMs.

    A minimum span prevents tiny numerical differences from being visually
    exaggerated. Padding includes every error bar without leaving large empty
    regions. Error metrics can be anchored at zero, while correlations and
    bounded scores retain their meaningful domains.
    """
    if summary.empty:
        return None
    means = summary["mean"].to_numpy(dtype=float)
    sems = summary["sem"].fillna(0.0).to_numpy(dtype=float)
    finite = np.isfinite(means) & np.isfinite(sems)
    if not finite.any():
        return None
    observed_low = float(np.min(means[finite] - sems[finite]))
    observed_high = float(np.max(means[finite] + sems[finite]))
    observed_span = max(observed_high - observed_low, 1e-9)
    if anchor_zero:
        lower = 0.0
        upper = max(
            observed_high + padding_fraction * observed_span,
            float(minimum_span),
        )
    else:
        requested_span = max(
            observed_span * (1.0 + 2.0 * padding_fraction),
            float(minimum_span),
        )
        center = 0.5 * (observed_low + observed_high)
        lower = center - 0.5 * requested_span
        upper = center + 0.5 * requested_span
    if bounds is not None:
        lower_bound, upper_bound = bounds
        if lower_bound is not None and lower < lower_bound:
            upper += float(lower_bound) - lower
            lower = float(lower_bound)
        if upper_bound is not None and upper > upper_bound:
            lower -= upper - float(upper_bound)
            upper = float(upper_bound)
        if lower_bound is not None:
            lower = max(lower, float(lower_bound))
        if upper_bound is not None:
            upper = min(upper, float(upper_bound))
    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
        return None
    return lower, upper


def save_focused_method_legend(
    methods: Sequence[str],
    output_path: Path,
    dpi: int,
) -> None:
    order = publication_method_order(methods, include_full_cells=True)
    handles = [
        Line2D(
            [0],
            [0],
            color=PUBLICATION_PALETTE.get(method, "#555555"),
            marker="o",
            markersize=7.0,
            linewidth=2.2,
            label=method,
        )
        for method in order
    ]
    fig = plt.figure(figsize=(max(10.5, 1.72 * len(order)), 0.86))
    legend = fig.legend(
        handles=handles,
        labels=order,
        loc="center",
        ncol=len(order),
        frameon=False,
        handlelength=1.7,
        handletextpad=0.36,
        columnspacing=0.88,
        borderaxespad=0,
        fontsize=16.0,
    )
    safe_mkdir(output_path.parent)
    fig.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
        bbox_extra_artists=(legend,),
        pad_inches=0.005,
        facecolor="white",
    )
    fig.savefig(
        output_path.with_suffix(".pdf"),
        bbox_inches="tight",
        bbox_extra_artists=(legend,),
        pad_inches=0.005,
        facecolor="white",
    )
    plt.close(fig)


def plot_publication_boxplot(
    frame: pd.DataFrame,
    value_column: str,
    ylabel: str,
    output_path: Path,
    dpi: int,
) -> None:
    work = publication_method_frame(frame)
    order = publication_method_order(work["method"], include_full_cells=False)
    # These panels are used at full supplementary-page width.  A compact
    # source canvas prevents LaTeX from shrinking otherwise large nominal
    # font sizes back to illegible 4--5 pt labels.
    fig_width = 9.6 if len(order) > 6 else 5.4
    fig, ax = plt.subplots(figsize=(fig_width, 5.0))
    sns.boxplot(
        data=work,
        x="method",
        y=value_column,
        order=order,
        hue="method",
        hue_order=order,
        palette=PUBLICATION_PALETTE,
        legend=False,
        width=0.68,
        linewidth=1.8,
        fliersize=0,
        ax=ax,
    )
    sns.stripplot(
        data=work,
        x="method",
        y=value_column,
        order=order,
        color="#222222",
        size=5.4,
        jitter=0.16,
        alpha=0.56,
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel(ylabel, fontsize=DE_AXIS_LABEL_FONTSIZE)
    ax.tick_params(axis="x", rotation=28, labelsize=DE_TICK_FONTSIZE)
    ax.tick_params(axis="y", labelsize=DE_TICK_FONTSIZE)
    sns.despine(ax=ax)
    fig.tight_layout()
    save_publication_figure(fig, output_path, dpi)


def plot_publication_sensitivity(
    frame: pd.DataFrame,
    x_column: str,
    value_column: str,
    ylabel: str,
    output_path: Path,
    dpi: int,
    include_full_cells: bool = False,
    x_tick_suffix: str = "",
    y_limits: Optional[Tuple[float, float]] = None,
    method_subset: Optional[Sequence[str]] = None,
    minimum_y_span: float = 0.0,
    y_bounds: Optional[Tuple[Optional[float], Optional[float]]] = None,
    anchor_zero: bool = False,
) -> None:
    work = publication_method_frame(frame)
    if method_subset is not None:
        requested = [publication_method_label(method) for method in method_subset]
        work = work[work["method"].isin(requested)].copy()
        if work.empty:
            raise ValueError(
                "None of the requested methods are present in the sensitivity table: "
                f"{requested}"
            )
    order = publication_method_order(
        work["method"], include_full_cells=include_full_cells
    )
    summary = summarize_mean_sem(work, [x_column, "method"], value_column)
    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    for method in order:
        method_frame = summary[summary["method"] == method].sort_values(x_column)
        if method_frame.empty:
            continue
        ax.errorbar(
            method_frame[x_column],
            method_frame["mean"],
            yerr=method_frame["sem"],
            color=PUBLICATION_PALETTE.get(method, "#555555"),
            marker="o",
            markersize=7.0,
            linewidth=2.4,
            capsize=3.4,
            capthick=1.3,
            label=method,
        )
    ticks = sorted(work[x_column].dropna().unique())
    ax.set_xticks(ticks)
    if x_tick_suffix:
        ax.set_xticklabels([f"{value:g}{x_tick_suffix}" for value in ticks])
    ax.set_xlabel("Compression" if x_tick_suffix else "Requested metacells")
    ax.set_ylabel(ylabel)
    ax.xaxis.label.set_size(DE_AXIS_LABEL_FONTSIZE)
    ax.yaxis.label.set_size(DE_AXIS_LABEL_FONTSIZE)
    ax.tick_params(axis="both", labelsize=DE_TICK_FONTSIZE)
    resolved_y_limits = y_limits
    if resolved_y_limits is None:
        resolved_y_limits = data_driven_y_limits(
            summary,
            minimum_span=minimum_y_span,
            bounds=y_bounds,
            anchor_zero=anchor_zero,
        )
    if resolved_y_limits is not None:
        ax.set_ylim(*resolved_y_limits)
        logger.info(
            "Resolved y-axis | panel=%s limits=(%.6f, %.6f)",
            output_path.name,
            resolved_y_limits[0],
            resolved_y_limits[1],
        )
    sns.despine(ax=ax)
    fig.tight_layout()
    save_publication_figure(fig, output_path, dpi)


# Reference-defined examples for the condition-DE heatmap.  Candidate pairs
# were selected without examining any metacell method: full-cell FDR < 0.05 and
# |logFC| >= 0.25, intersected with the external MSigDB Hallmark interferon
# alpha/gamma programs and restricted to the major immune populations reported
# in the source COVID-19 study.  Candidates were ranked by full-cell |logFC|
# and then |Wilcoxon score|, with unique genes and cell types.  The resulting
# fixed list makes the main and supplementary figures reproducible while
# avoiding selection on CAMP1 performance.
CONDITION_DE_EXAMPLE_PAIRS: List[Tuple[str, str, str]] = [
    ("CD16 Monocyte", "IFI27", "CD16 Monocyte | IFI27"),
    ("NK", "IFI44L", "NK | IFI44L"),
    ("DC", "IFITM3", "DC | IFITM3"),
    ("CD4n T", "SOCS3", "CD4n T | SOCS3"),
    ("CD14 Monocyte", "PIM1", "CD14 Monocyte | PIM1"),
    ("CD8eff T", "IFIT3", "CD8eff T | IFIT3"),
]


def read_condition_de_examples(rate_dir: Path) -> pd.DataFrame:
    """Read curated, biologically interpretable COVID-response examples."""
    file_methods = [
        ("original", "Full cells"),
        ("camp1", "CAMP1"),
        ("camp2", "CAMP2"),
        ("camp3", "CAMP3"),
        ("camp4", "CAMP4"),
        ("seacells", "SEACells"),
        ("supercell", "SuperCell"),
        ("metacell1", "MetaCell"),
        ("metacell2", "MetaCell2"),
        ("metaq", "MetaQ"),
    ]
    pair_order = {
        (celltype, gene): (index, label)
        for index, (celltype, gene, label) in enumerate(CONDITION_DE_EXAMPLE_PAIRS)
    }
    frames: List[pd.DataFrame] = []
    for file_stem, method in file_methods:
        path = rate_dir / f"{file_stem}_condition_de.csv.gz"
        if not path.is_file():
            raise FileNotFoundError(f"Missing matched condition-DE table: {path}")
        frame = pd.read_csv(
            path,
            usecols=[
                "names",
                "logfoldchanges",
                "pvals_adj",
                "celltype",
                "contrast",
                "reference",
            ],
        )
        frame = frame[
            frame.apply(
                lambda row: (str(row["celltype"]), str(row["names"])) in pair_order,
                axis=1,
            )
        ].copy()
        frame["method"] = method
        frame["example_order"] = frame.apply(
            lambda row: pair_order[(str(row["celltype"]), str(row["names"]))][0],
            axis=1,
        )
        frame["example"] = frame.apply(
            lambda row: pair_order[(str(row["celltype"]), str(row["names"]))][1],
            axis=1,
        )
        frames.append(frame)
    examples = pd.concat(frames, ignore_index=True)
    expected_rows = len(CONDITION_DE_EXAMPLE_PAIRS) * len(file_methods)
    if len(examples) != expected_rows:
        observed = examples.groupby("method").size().to_dict()
        raise ValueError(
            "Condition-DE example table is incomplete; "
            f"expected {expected_rows} rows, observed {observed}"
        )
    return examples.sort_values(["example_order", "method"]).reset_index(drop=True)


def condition_de_example_color_limit(examples: pd.DataFrame) -> float:
    """Return a symmetric half-unit color limit that contains every value."""
    maximum = pd.to_numeric(examples["logfoldchanges"], errors="coerce").abs().max()
    if not np.isfinite(maximum):
        raise ValueError("Condition-DE examples contain no finite log-fold changes")
    return max(1.0, float(np.ceil(float(maximum) * 2.0) / 2.0))


def plot_condition_de_example_heatmap(
    examples: pd.DataFrame,
    method_order: Sequence[str],
    output_path: Path,
    dpi: int,
    color_limit: float,
) -> None:
    """Plot representative condition effects with numeric values and audit boxes."""
    work = examples[examples["method"].isin(method_order)].copy()
    row_order = [label for _, _, label in CONDITION_DE_EXAMPLE_PAIRS]
    matrix = work.pivot(index="example", columns="method", values="logfoldchanges")
    matrix = matrix.reindex(index=row_order, columns=list(method_order))
    if matrix.isna().any().any():
        raise ValueError("Condition-DE example heatmap contains missing values")

    width = 9.4 if len(method_order) <= 7 else 13.2
    fig, ax = plt.subplots(figsize=(width, 6.8))
    sns.heatmap(
        matrix,
        cmap="RdBu_r",
        center=0.0,
        vmin=-float(color_limit),
        vmax=float(color_limit),
        cbar=True,
        cbar_kws={
            "label": "COVID vs Healthy\nlog fold change",
            "ticks": [-float(color_limit), 0.0, float(color_limit)],
            "shrink": 0.82,
            "pad": 0.025,
        },
        linewidths=0.7,
        linecolor="white",
        square=False,
        ax=ax,
    )
    for row_index, example in enumerate(matrix.index):
        full_value = float(matrix.loc[example, "Full cells"])
        camp_error = abs(float(matrix.loc[example, "CAMP1"]) - full_value)
        for column_index, method in enumerate(matrix.columns):
            value = float(matrix.loc[example, method])
            text_color = "white" if abs(value) >= 0.52 * color_limit else "#222222"
            ax.text(
                column_index + 0.5,
                row_index + 0.5,
                f"{value:.2f}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=DE_HEATMAP_VALUE_FONTSIZE,
            )
            # A green outline has an objective meaning: the comparator differs
            # from the full-cell effect by at least 0.5 log-fold-change units
            # in a row where CAMP1 is within 0.2 units of the reference.
            if (
                method in DE_EXAMPLE_EXTERNAL_COMPARATORS
                and camp_error <= 0.2
                and abs(value - full_value) >= 0.5
            ):
                ax.add_patch(
                    Rectangle(
                        (column_index, row_index),
                        1,
                        1,
                        fill=False,
                        edgecolor=DE_EXAMPLE_HIGHLIGHT_COLOR,
                        linewidth=4.2,
                    )
                )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticklabels(
        ax.get_xticklabels(), rotation=30, ha="right", fontsize=DE_TICK_FONTSIZE
    )
    ax.set_yticklabels(
        ax.get_yticklabels(), rotation=0, fontsize=DE_TICK_FONTSIZE + 1
    )
    ax.tick_params(axis="both", length=0, pad=6)
    colorbar = ax.collections[0].colorbar
    colorbar.ax.tick_params(labelsize=DE_TICK_FONTSIZE, length=4)
    colorbar.ax.yaxis.label.set_size(DE_AXIS_LABEL_FONTSIZE)
    fig.tight_layout()
    save_publication_figure(fig, output_path, dpi)


def save_condition_de_colorbar(
    output_path: Path,
    dpi: int,
    color_limit: float,
) -> None:
    """Save the shared frameless horizontal log-fold-change color scale."""
    norm = matplotlib.colors.Normalize(vmin=-float(color_limit), vmax=float(color_limit))
    scalar = matplotlib.cm.ScalarMappable(norm=norm, cmap="RdBu_r")
    scalar.set_array([])
    fig = plt.figure(figsize=(4.6, 0.62))
    colorbar_axis = fig.add_axes([0.035, 0.58, 0.93, 0.20])
    colorbar = fig.colorbar(scalar, cax=colorbar_axis, orientation="horizontal")
    colorbar.outline.set_visible(False)
    colorbar.set_ticks([-float(color_limit), 0.0, float(color_limit)])
    colorbar.ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.1f"))
    colorbar.ax.tick_params(axis="x", labelsize=6.5, length=2.0, pad=1.0)
    colorbar.set_label(
        "COVID vs Healthy log fold change",
        fontsize=7.5,
        labelpad=1.5,
    )
    safe_mkdir(output_path.parent)
    for destination in [output_path, output_path.with_suffix(".pdf")]:
        fig.savefig(
            destination,
            dpi=dpi if destination.suffix == ".png" else None,
            bbox_inches="tight",
            pad_inches=0.003,
            facecolor="white",
        )
    plt.close(fig)


def compute_de_evaluation_umap_from_pbmc_pca(output_root: Path) -> pd.DataFrame:
    """Reproduce the CAMP PBMC UMAP on the shared DE evaluation universe.

    This intentionally follows the reference notebook's effective sequence:
    construct a 15-neighbor graph from ``X_pca`` and run UMAP with Scanpy's
    defaults (two dimensions, min_dist=0.5, spread=1, random_state=0). Cell-type
    and clinical-status panels are recolorings of these identical coordinates.
    """
    cache_path = output_root / "cache" / "pbmc_hvg2000_pca50.h5ad"
    if not cache_path.is_file():
        raise FileNotFoundError(
            "The preprocessed PBMC PCA cache is required for the status UMAP: "
            f"{cache_path}"
        )

    common_path = (
        output_root
        / "generated_partitions"
        / "native_resolution_grid"
        / "released_baselines"
        / "common_evaluation_cell_ids.csv.gz"
    )
    if not common_path.is_file():
        raise FileNotFoundError(
            "The shared all-method PBMC evaluation IDs are required for the "
            f"DE UMAP panels: {common_path}"
        )

    reference = ad.read_h5ad(cache_path)
    if "X_pca" not in reference.obsm:
        raise KeyError(f"X_pca is missing from the PBMC cache: {cache_path}")
    if "Status" not in reference.obs:
        raise KeyError(f"Status is missing from the PBMC cache: {cache_path}")

    common_frame = pd.read_csv(common_path)
    if "cell_id" not in common_frame:
        raise KeyError(f"cell_id is missing from {common_path}")
    common_ids = pd.Index(common_frame["cell_id"].astype(str))
    positions = pd.Index(reference.obs_names.astype(str)).get_indexer(common_ids)
    if np.any(positions < 0):
        missing = common_ids[np.flatnonzero(positions < 0)].tolist()[:10]
        raise KeyError(
            "Shared evaluation cells are absent from the PBMC PCA cache; "
            f"examples: {missing}"
        )
    evaluation = reference[positions].copy()

    # umap-learn performs the same PCA-neighbor-to-UMAP operation used by
    # sc.pp.neighbors(..., use_rep='X_pca'); sc.tl.umap(...).  Calling it
    # directly keeps the plotting stage self-contained and deterministic.
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
    coordinates = reducer.fit_transform(np.asarray(evaluation.obsm["X_pca"]))
    result = pd.DataFrame(
        {
            "item_id": evaluation.obs_names.astype(str),
            "condition": evaluation.obs["Status"].astype(str).to_numpy(),
            "celltype": evaluation.obs["cell.type"].astype(str).to_numpy(),
            "donor": evaluation.obs["Donor"].astype(str).to_numpy(),
            "UMAP1": coordinates[:, 0],
            "UMAP2": coordinates[:, 1],
        }
    )
    result["embedding_source"] = (
        "common_evaluation_X_pca50_umap_n15_min_dist0.5_random_state0"
    )
    return result


def compute_full_pbmc_status_umap_from_pca(output_root: Path) -> pd.DataFrame:
    """Compute the reference-shaped UMAP on all 37,582 post-filtering cells."""
    cache_path = output_root / "cache" / "pbmc_hvg2000_pca50.h5ad"
    if not cache_path.is_file():
        raise FileNotFoundError(
            "The preprocessed PBMC PCA cache is required for the full status UMAP: "
            f"{cache_path}"
        )
    reference = ad.read_h5ad(cache_path)
    if "X_pca" not in reference.obsm:
        raise KeyError(f"X_pca is missing from the PBMC cache: {cache_path}")
    for key in ["Status", "cell.type", "Donor"]:
        if key not in reference.obs:
            raise KeyError(f"{key} is missing from the PBMC cache: {cache_path}")

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
    result["embedding_source"] = (
        "all_37582_postfilter_X_pca50_umap_n15_min_dist0.5_random_state0"
    )
    return result


def plot_celltype_umap_for_de(
    frame: pd.DataFrame,
    output_path: Path,
    dpi: int,
) -> None:
    """Plot the shared DE evaluation UMAP colored by annotated cell type."""
    required = {"UMAP1", "UMAP2", "celltype"}
    if frame.empty or not required.issubset(frame.columns):
        raise ValueError(f"Cell-type UMAP needs columns {sorted(required)}")
    work = frame.copy()
    work["celltype"] = work["celltype"].astype(str)
    order = list(dict.fromkeys(work["celltype"].tolist()))
    palette_values = sns.color_palette("tab20", n_colors=len(order))
    palette = dict(zip(order, palette_values))
    work = work.sample(frac=1.0, random_state=0)

    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    fig.subplots_adjust(left=0.01, right=0.69, bottom=0.025, top=0.975)
    ax.scatter(
        work["UMAP1"],
        work["UMAP2"],
        s=2.8,
        alpha=0.76,
        linewidths=0,
        rasterized=True,
        c=work["celltype"].map(palette),
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
            markerfacecolor=palette[celltype],
            markeredgecolor="none",
            markersize=5.4,
            label=celltype,
        )
        for celltype in order
    ]
    ax.legend(
        handles=handles,
        title="Cell type",
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        ncol=1,
        frameon=False,
        fontsize=12.5,
        title_fontsize=14,
        handletextpad=0.45,
        labelspacing=0.28,
        borderaxespad=0,
    )
    save_publication_figure(fig, output_path, dpi)


def plot_status_umap_for_de(
    frame: pd.DataFrame,
    output_path: Path,
    dpi: int,
    legend_position: str = "bottom",
) -> None:
    """Show the full-cell COVID/Healthy contrast used by condition DE."""
    required = {"UMAP1", "UMAP2", "condition"}
    if frame.empty or not required.issubset(frame.columns):
        raise ValueError(f"Status UMAP needs columns {sorted(required)}")
    work = frame.copy()
    work["condition"] = work["condition"].astype(str)
    order = [value for value in ["Healthy", "COVID"] if value in set(work["condition"])]
    order.extend(sorted(set(work["condition"]).difference(order)))
    work = work.sample(frac=1.0, random_state=0)
    if legend_position == "right":
        fig, ax = plt.subplots(figsize=(6.2, 4.35))
        fig.subplots_adjust(left=0.01, right=0.75, bottom=0.025, top=0.975)
    elif legend_position == "bottom":
        fig, ax = plt.subplots(figsize=(5.35, 5.15))
        fig.subplots_adjust(left=0.015, right=0.985, bottom=0.155, top=0.985)
    else:
        raise ValueError("legend_position must be 'bottom' or 'right'")
    ax.scatter(
        work["UMAP1"],
        work["UMAP2"],
        s=3.0,
        alpha=0.72,
        linewidths=0,
        rasterized=True,
        c=work["condition"].map(DE_STATUS_PALETTE).fillna("#777777"),
    )
    ax.set_aspect("equal", adjustable="box")
    ax.margins(x=0.01, y=0.01)
    ax.axis("off")
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=DE_STATUS_PALETTE.get(condition, "#777777"),
            markeredgecolor="none",
            markersize=6.5,
            label=condition,
        )
        for condition in order
    ]
    legend_kwargs = (
        {
            "loc": "center left",
            "bbox_to_anchor": (1.01, 0.5),
            "ncol": 1,
            "columnspacing": 1.0,
            "labelspacing": 0.65,
        }
        if legend_position == "right"
        else {
            "loc": "upper center",
            "bbox_to_anchor": (0.5, -0.018),
            "ncol": len(legend_handles),
            "columnspacing": 1.3,
            "labelspacing": 0.2,
        }
    )
    legend = ax.legend(
        handles=legend_handles,
        title="Status",
        frameon=False,
        fontsize=DE_TICK_FONTSIZE,
        title_fontsize=DE_TICK_FONTSIZE + 1,
        markerscale=1.3,
        handletextpad=0.55,
        borderaxespad=0,
        **legend_kwargs,
    )
    for handle in legend.legend_handles:
        handle.set_alpha(1.0)
    save_publication_figure(fig, output_path, dpi)


def plot_de_rank_consistency_boxplot(
    concordance: pd.DataFrame,
    output_path: Path,
    dpi: int,
    method_order: Sequence[str] = DEFAULT_FOCUSED_METHOD_ORDER,
) -> None:
    """MetaQ-style all-comparator boxplot across PBMC cell types."""
    work = publication_method_frame(concordance)
    requested = [publication_method_label(method) for method in method_order]
    work = work[work["method"].isin(requested)].copy()
    order = [method for method in requested if method in set(work["method"])]
    if not order:
        raise ValueError("No requested methods are present in the DE-rank table")
    # Focused boxplots are placed at 35% of the manuscript width.  This
    # compact canvas yields approximately 8--10 pt labels in the composite.
    fig, ax = plt.subplots(figsize=(5.4, 4.8))
    sns.boxplot(
        data=work,
        x="method",
        y="kendall_tau",
        order=order,
        hue="method",
        hue_order=order,
        palette=PUBLICATION_PALETTE,
        legend=False,
        width=0.66,
        linewidth=1.8,
        saturation=0.78,
        fliersize=0,
        ax=ax,
    )
    sns.stripplot(
        data=work,
        x="method",
        y="kendall_tau",
        order=order,
        color="#222222",
        size=5.6,
        jitter=0.16,
        alpha=0.58,
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("Kendall's tau", fontsize=DE_AXIS_LABEL_FONTSIZE)
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels(order, rotation=25, ha="right", fontsize=DE_TICK_FONTSIZE)
    ax.tick_params(axis="y", labelsize=DE_TICK_FONTSIZE)
    observed_low = float(work["kendall_tau"].min())
    observed_high = float(work["kendall_tau"].max())
    margin = max(0.012, 0.08 * (observed_high - observed_low))
    ax.set_ylim(observed_low - margin, min(1.0, observed_high + margin))
    sns.despine(ax=ax)
    fig.tight_layout()
    save_publication_figure(fig, output_path, dpi)


def plot_condition_de_consistency_boxplot(
    correlations: pd.DataFrame,
    output_path: Path,
    dpi: int,
    method_order: Sequence[str] = DEFAULT_FOCUSED_METHOD_ORDER,
) -> None:
    """MetaQ-style condition-DE consistency at the prespecified 10x rate."""
    work = publication_method_frame(correlations)
    requested = [publication_method_label(method) for method in method_order]
    work = work[work["method"].isin(requested)].copy()
    order = [method for method in requested if method in set(work["method"])]
    if not order:
        raise ValueError("No requested methods are present in the condition-DE table")
    fig, ax = plt.subplots(figsize=(5.4, 4.8))
    sns.boxplot(
        data=work,
        x="method",
        y="pearson_r",
        order=order,
        hue="method",
        hue_order=order,
        palette=PUBLICATION_PALETTE,
        legend=False,
        width=0.66,
        linewidth=1.8,
        saturation=0.78,
        fliersize=0,
        ax=ax,
    )
    sns.stripplot(
        data=work,
        x="method",
        y="pearson_r",
        order=order,
        color="#222222",
        size=5.6,
        jitter=0.16,
        alpha=0.58,
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("Pearson correlation", fontsize=DE_AXIS_LABEL_FONTSIZE)
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels(order, rotation=25, ha="right", fontsize=DE_TICK_FONTSIZE)
    ax.tick_params(axis="y", labelsize=DE_TICK_FONTSIZE)
    observed_low = float(work["pearson_r"].min())
    observed_high = float(work["pearson_r"].max())
    margin = max(0.004, 0.08 * (observed_high - observed_low))
    ax.set_ylim(max(-1.0, observed_low - margin), min(1.0, observed_high + margin))
    sns.despine(ax=ax)
    fig.tight_layout()
    save_publication_figure(fig, output_path, dpi)


def copy_publication_pair(source_stem: Path, destination_stem: Path) -> None:
    """Copy a rendered PNG/PDF pair while preserving the canonical source."""
    for suffix in [".png", ".pdf"]:
        source = source_stem.with_suffix(suffix)
        if not source.is_file():
            raise FileNotFoundError(f"Missing rendered figure: {source}")
        destination = destination_stem.with_suffix(suffix)
        safe_mkdir(destination.parent)
        shutil.copy2(source, destination)


def plot_de_only_manuscript_outputs(
    args: argparse.Namespace,
    output_root: Path,
) -> None:
    """Render the final DE-only manuscript set without B-lineage dependencies."""
    output_dir = output_root / "figures" / "manuscript_additional_experiments_matched_v4"
    main_dir = output_dir / "main_text"
    supplement_dir = output_dir / "supplement"
    figure_data_dir = output_dir / "figure_data"
    for path in [output_dir, main_dir, supplement_dir, figure_data_dir]:
        safe_mkdir(path)

    # Cell-type and status views use identical UMAP coordinates computed from
    # the same 33,320-cell universe used by every quantitative DE comparison.
    status_umap_frame = compute_de_evaluation_umap_from_pbmc_pca(output_root)
    plot_celltype_umap_for_de(
        status_umap_frame,
        main_dir / "figA_celltype_umap.png",
        args.dpi,
    )
    plot_status_umap_for_de(
        status_umap_frame,
        main_dir / "figA_status_umap.png",
        args.dpi,
    )
    save_csv(
        status_umap_frame,
        figure_data_dir / "figA_status_umap_coordinates.csv.gz",
    )

    # Keep a separate full-cohort context panel with the geometry of the
    # original CAMP PBMC notebook. It is never substituted for the 33,320-cell
    # quantitative evaluation universe.
    full_status_umap_frame = compute_full_pbmc_status_umap_from_pca(output_root)
    plot_status_umap_for_de(
        full_status_umap_frame,
        main_dir / "figA_status_umap_all_cells.png",
        args.dpi,
        legend_position="right",
    )
    # Preserve the right-legend panel and additionally provide the identical
    # all-cell embedding with a compact horizontal legend underneath.  Both
    # files use the same coordinates, so their orientation and point geometry
    # are exactly the same.
    plot_status_umap_for_de(
        full_status_umap_frame,
        main_dir / "figA_status_umap_all_cells_legend_below.png",
        args.dpi,
        legend_position="bottom",
    )
    save_csv(
        full_status_umap_frame,
        figure_data_dir / "figA_status_umap_all_cells_coordinates.csv.gz",
    )

    fig5_root = (
        output_root
        / "intermediate_csv"
        / "fig5_de_preservation"
        / "matched_reduction_rates"
    )
    concordance_frames: List[pd.DataFrame] = []
    correlation_frames: List[pd.DataFrame] = []
    for rate in args.de_reduction_rates:
        rate_dir = fig5_root / reduction_rate_slug(float(rate))
        concordance = read_csvs(rate_dir.glob("*_celltype_rank_concordance.csv"))
        correlations = read_csvs(rate_dir.glob("*_condition_correlations.csv"))
        if concordance.empty or correlations.empty:
            raise FileNotFoundError(f"Incomplete matched DE tables: {rate_dir}")
        concordance["compression"] = float(rate)
        correlations["compression"] = float(rate)
        concordance_frames.append(concordance)
        correlation_frames.append(correlations)
    concordance_all = pd.concat(concordance_frames, ignore_index=True)
    correlations_all = pd.concat(correlation_frames, ignore_index=True)
    primary_rate = float(args.de_primary_reduction_rate)
    primary_concordance = concordance_all[
        np.isclose(concordance_all["compression"], primary_rate)
    ].copy()
    primary_correlations = correlations_all[
        np.isclose(correlations_all["compression"], primary_rate)
    ].copy()
    primary_rate_dir = fig5_root / reduction_rate_slug(primary_rate)

    metaq_fig5_meta_dir = (
        output_root
        / "intermediate_csv"
        / "metaq_paper_style_matched_counts"
        / "fig5"
    )
    logfc_long = pd.read_csv(metaq_fig5_meta_dir / "fig5b_logfoldchanges.csv.gz")
    selected_celltypes = (
        pd.read_csv(metaq_fig5_meta_dir / "fig5_selected_celltypes.csv")
        .sort_values("selection_order")["celltype"]
        .astype(str)
        .tolist()
    )
    selected_genes = (
        pd.read_csv(metaq_fig5_meta_dir / "fig5_selected_genes.csv")
        .sort_values("selection_order")["gene"]
        .astype(str)
        .tolist()
    )
    rank_matrices: Dict[str, pd.DataFrame] = {}
    for method, method_frame in logfc_long.groupby("method", sort=False):
        matrix = method_frame.pivot(
            index="celltype", columns="gene", values="logfoldchanges"
        ).reindex(index=selected_celltypes, columns=selected_genes)
        rank_matrices[str(method)] = logfc_to_rank_matrix(matrix)
    if "Original cells" not in rank_matrices or "CAMP1" not in rank_matrices:
        raise ValueError("The DE-rank matrices must contain Original cells and CAMP1")
    canonical_rank_stem = main_dir / "figB_de_rank_heatmap_camp1"
    plot_metaq_rank_heatmap_pair(
        rank_matrices["Original cells"],
        rank_matrices["CAMP1"],
        "CAMP1",
        canonical_rank_stem.with_suffix(".png"),
        args.dpi,
        show_colorbar=False,
    )
    rank_colorbar_stem = main_dir / "figB_de_rank_colorbar_horizontal"
    save_de_rank_colorbar(
        rank_colorbar_stem.with_suffix(".png"),
        args.dpi,
        maximum_rank=rank_matrices["Original cells"].shape[1],
    )
    copy_publication_pair(
        canonical_rank_stem,
        main_dir / "figA_de_rank_heatmap_camp1",
    )

    plot_de_rank_consistency_boxplot(
        primary_concordance,
        main_dir / "figC_de_rank_consistency_boxplot.png",
        args.dpi,
    )
    save_csv(
        publication_method_frame(primary_concordance)[
            publication_method_frame(primary_concordance)["method"].isin(
                DEFAULT_FOCUSED_METHOD_ORDER
            )
        ],
        figure_data_dir / "figC_de_rank_consistency_boxplot.csv",
    )

    condition_examples = read_condition_de_examples(primary_rate_dir)
    condition_color_limit = condition_de_example_color_limit(condition_examples)
    main_example_methods = ["Full cells", *DEFAULT_FOCUSED_METHOD_ORDER]
    canonical_example_stem = main_dir / "figD_de_condition_gene_examples_camp1"
    plot_condition_de_example_heatmap(
        condition_examples,
        main_example_methods,
        canonical_example_stem.with_suffix(".png"),
        args.dpi,
        color_limit=condition_color_limit,
    )
    copy_publication_pair(
        canonical_example_stem,
        main_dir / "figB_de_condition_gene_examples_camp1",
    )
    save_csv(
        condition_examples[condition_examples["method"].isin(main_example_methods)],
        figure_data_dir / "figD_de_condition_gene_examples_camp1.csv",
    )

    plot_condition_de_consistency_boxplot(
        primary_correlations,
        main_dir / "figE_de_condition_consistency_boxplot.png",
        args.dpi,
    )
    save_csv(
        publication_method_frame(primary_correlations)[
            publication_method_frame(primary_correlations)["method"].isin(
                DEFAULT_FOCUSED_METHOD_ORDER
            )
        ],
        figure_data_dir / "figE_de_condition_consistency_boxplot.csv",
    )

    # MetaQ evaluates DE at a fixed 10x reduction.  The three-rate line plot is
    # a CAMP robustness extension, so keep it as a supplementary result rather
    # than presenting it as a direct MetaQ Fig. 5 counterpart.
    canonical_sensitivity_stem = (
        supplement_dir / "figS_de_condition_sensitivity_default_camp1"
    )
    plot_publication_sensitivity(
        correlations_all,
        "compression",
        "pearson_r",
        "Mean Pearson correlation",
        canonical_sensitivity_stem.with_suffix(".png"),
        args.dpi,
        x_tick_suffix="x",
        method_subset=DEFAULT_FOCUSED_METHOD_ORDER,
        minimum_y_span=0.012,
        y_bounds=(0.0, 1.0),
    )
    copy_publication_pair(
        canonical_sensitivity_stem,
        main_dir / "figC_de_condition_sensitivity",
    )
    save_csv(
        publication_method_frame(correlations_all)[
            publication_method_frame(correlations_all)["method"].isin(
                DEFAULT_FOCUSED_METHOD_ORDER
            )
        ],
        figure_data_dir / "figS_de_condition_sensitivity_default_camp1.csv",
    )
    save_focused_method_legend(
        DEFAULT_FOCUSED_METHOD_ORDER,
        main_dir / "shared_method_legend_default_camp1.png",
        args.dpi,
    )

    # Curate a clean DE-only delivery folder.  This deliberately excludes the
    # archived batch-integration and B-lineage panels that remain elsewhere in
    # the v4 directory for provenance.
    final_root = output_dir / "de_only_final"
    final_main_dir = final_root / "main_text"
    final_supplement_dir = final_root / "supplement"
    for path in [final_root, final_main_dir, final_supplement_dir]:
        safe_mkdir(path)

    final_main_stems = [
        "figA_celltype_umap",
        "figA_status_umap",
        "figA_status_umap_all_cells",
        "figA_status_umap_all_cells_legend_below",
        "figB_de_rank_heatmap_camp1",
        "figB_de_rank_colorbar_horizontal",
        "figC_de_rank_consistency_boxplot",
        "figD_de_condition_gene_examples_camp1",
        "figE_de_condition_consistency_boxplot",
    ]
    for stem in final_main_stems:
        copy_publication_pair(main_dir / stem, final_main_dir / stem)

    plot_metaq_rank_heatmap_grid(
        rank_matrices,
        ["Original cells", *ALL_METHOD_ORDER],
        final_supplement_dir / "figS01_de_rank_heatmap_grid_all_methods.png",
        args.dpi,
    )
    plot_condition_de_example_heatmap(
        condition_examples,
        ["Full cells", *PUBLICATION_METHOD_ORDER],
        final_supplement_dir / "figS02_de_condition_gene_examples_all_methods.png",
        args.dpi,
        color_limit=condition_color_limit,
    )
    copy_publication_pair(
        canonical_sensitivity_stem,
        final_supplement_dir / "figS03_de_condition_sensitivity_default_camp1",
    )
    copy_publication_pair(
        main_dir / "shared_method_legend_default_camp1",
        final_supplement_dir / "figS04_methods_legend_default_camp1",
    )
    plot_publication_sensitivity(
        correlations_all,
        "compression",
        "pearson_r",
        "Mean Pearson correlation",
        final_supplement_dir / "figS05_de_condition_sensitivity_all_methods.png",
        args.dpi,
        x_tick_suffix="x",
        method_subset=PUBLICATION_METHOD_ORDER,
        minimum_y_span=0.012,
        y_bounds=(0.0, 1.0),
    )
    plot_publication_sensitivity(
        concordance_all,
        "compression",
        "kendall_tau",
        "Mean Kendall tau",
        final_supplement_dir / "figS06_de_celltype_sensitivity_all_methods.png",
        args.dpi,
        x_tick_suffix="x",
        method_subset=PUBLICATION_METHOD_ORDER,
        minimum_y_span=0.10,
        y_bounds=(-1.0, 1.0),
    )
    plot_publication_boxplot(
        primary_concordance,
        "kendall_tau",
        "Kendall's tau",
        final_supplement_dir / "figS07_de_celltype_kendall_m10_all_methods.png",
        args.dpi,
    )
    plot_publication_boxplot(
        primary_correlations,
        "pearson_r",
        "Pearson correlation",
        final_supplement_dir / "figS08_de_condition_pearson_m10_all_methods.png",
        args.dpi,
    )
    save_focused_method_legend(
        PUBLICATION_METHOD_ORDER,
        final_supplement_dir / "figS09_methods_legend_all_methods.png",
        args.dpi,
    )
    save_json(
        {
            "protocol": "de_only_final_figure_package_v3",
            "main_text": final_main_stems,
            "supplement": [
                "figS01_de_rank_heatmap_grid_all_methods",
                "figS02_de_condition_gene_examples_all_methods",
                "figS03_de_condition_sensitivity_default_camp1",
                "figS04_methods_legend_default_camp1",
                "figS05_de_condition_sensitivity_all_methods",
                "figS06_de_celltype_sensitivity_all_methods",
                "figS07_de_celltype_kendall_m10_all_methods",
                "figS08_de_condition_pearson_m10_all_methods",
                "figS09_methods_legend_all_methods",
            ],
            "boxplot_internal_text": False,
            "large_font_layout": True,
            "status_palette": DE_STATUS_PALETTE,
            "status_umap_cells": 33320,
            "status_umap_source": (
                "shared all-method DE evaluation PBMC X_pca; UMAP n_neighbors=15, "
                "min_dist=0.5, random_state=0"
            ),
            "de_evaluation_cells": 33320,
            "optional_full_status_umap": {
                "stem": "figA_status_umap_all_cells",
                "bottom_legend_stem": "figA_status_umap_all_cells_legend_below",
                "cells": 37582,
                "purpose": "full post-filtering cohort context only",
            },
            "condition_example_outline_color": DE_EXAMPLE_HIGHLIGHT_COLOR,
            "condition_example_selection": (
                "method-independent Hallmark IFN-alpha/gamma rule: full-cell "
                "FDR < 0.05 and |logFC| >= 0.25 in major immune populations; "
                "ranked by full-cell |logFC| then |Wilcoxon score| with unique "
                "genes and cell types"
            ),
            "condition_example_selection_uses_method_performance": False,
            "condition_example_outline_rule": (
                "external-comparator absolute error >= 0.5 while CAMP1 absolute "
                "error <= 0.2"
            ),
            "b_lineage_included": False,
            "batch_integration_included": False,
        },
        final_root / "figure_manifest.json",
    )
    save_json(
        {
            "protocol": "de_only_manuscript_panels_v3",
            "source_data": "Wilk et al. COVID-19 PBMC atlas",
            "main_text_method_policy": (
                "Prespecified CAMP1 versus all five competing methods; CAMP2-4 "
                "remain in all-method supplementary analyses."
            ),
            "de_compression_rates": args.de_reduction_rates,
            "de_primary_compression": args.de_primary_reduction_rate,
            "main_text_panels": [
                "figA_celltype_umap",
                "figA_status_umap",
                "figB_de_rank_heatmap_camp1",
                "figC_de_rank_consistency_boxplot",
                "figD_de_condition_gene_examples_camp1",
                "figE_de_condition_consistency_boxplot",
            ],
            "supplementary_robustness_panel": (
                "figS_de_condition_sensitivity_default_camp1"
            ),
            "shared_rank_color_scale": True,
            "rank_colorbar_attached": False,
            "rank_colorbar_stem": "figB_de_rank_colorbar_horizontal",
            "condition_heatmap_colorbar_attached": True,
            "b_lineage_in_main_text": False,
        },
        output_dir / "figure_set_protocol.json",
    )
    logger.info("DE-only manuscript panels saved: %s", main_dir)


def plot_focused_manuscript_outputs(
    args: argparse.Namespace,
    output_root: Path,
) -> None:
    """Render default-CAMP1 main panels and complete all-method supplements."""
    output_dir = output_root / "figures" / "manuscript_additional_experiments_matched_v4"
    main_dir = output_dir / "main_text"
    supplement_dir = output_dir / "supplement"
    figure_data_dir = output_dir / "figure_data"
    blineage_umap_dir = supplement_dir / "b_lineage_umaps_all_methods"
    blineage_marker_dir = supplement_dir / "b_lineage_marker_maps_all_methods"
    standalone_legend_dir = supplement_dir / "standalone_legends"
    for path in [
        output_dir,
        main_dir,
        supplement_dir,
        figure_data_dir,
        blineage_umap_dir,
        blineage_marker_dir,
        standalone_legend_dir,
    ]:
        safe_mkdir(path)

    methods_for_legend: set[str] = set()
    audit_rows: List[Dict[str, object]] = []
    manifest_rows: List[Dict[str, object]] = []

    def copy_figure_pair(
        source_stem: Path,
        destination_stem: Path,
        category: str,
        method: str = "",
        marker: str = "",
    ) -> None:
        copied: Dict[str, str] = {}
        for suffix in [".pdf", ".png"]:
            source = source_stem.with_suffix(suffix)
            if not source.is_file():
                raise FileNotFoundError(f"Missing source figure: {source}")
            destination = destination_stem.with_suffix(suffix)
            safe_mkdir(destination.parent)
            shutil.copy2(source, destination)
            copied[suffix] = str(destination.relative_to(output_dir))
        manifest_rows.append(
            {
                "category": category,
                "method": (
                    publication_method_label(
                        "Harmony" if method.lower() == "harmony" else display_name(method)
                    )
                    if method
                    else ""
                ),
                "marker": marker,
                "pdf": copied[".pdf"],
                "png": copied[".png"],
            }
        )

    # Batch integration is retained as complete supplementary evidence. No
    # compressed method dominates both axes, so this diagnostic is not used as
    # a main-text superiority claim.
    fig4_root = (
        output_root / "intermediate_csv" / "fig4_batch_integration" / "matched_counts"
    )
    lisi_path = fig4_root / "fig4_matched_count_lisi_all.csv"
    if not lisi_path.is_file():
        raise FileNotFoundError(f"Missing matched Fig. 4 LISI table: {lisi_path}")
    lisi = publication_method_frame(pd.read_csv(lisi_path))
    lisi = lisi[
        lisi["target_metacells"].astype(int) == int(args.fig4_primary_target)
    ].copy()
    methods_for_legend.update(lisi["method"].astype(str))
    paired = lisi.pivot_table(
        index=["method", "celltype"], columns="metric", values="score"
    ).reset_index()
    batch_rows = []
    for method, method_frame in paired.groupby("method", sort=False):
        batch_rows.append(
            {
                "method": method,
                "n_celltypes": int(len(method_frame)),
                "ilisi_mean": method_frame["iLISI_normalized"].mean(),
                "ilisi_sem": method_frame["iLISI_normalized"].sem(),
                "one_minus_clisi_mean": method_frame[
                    "one_minus_cLISI_normalized"
                ].mean(),
                "one_minus_clisi_sem": method_frame[
                    "one_minus_cLISI_normalized"
                ].sem(),
            }
        )
    batch_summary = pd.DataFrame(batch_rows)
    save_csv(batch_summary, output_dir / "batch_tradeoff_summary_m1000.csv")
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for _, row in batch_summary.iterrows():
        method = str(row["method"])
        ax.errorbar(
            row["ilisi_mean"],
            row["one_minus_clisi_mean"],
            xerr=row["ilisi_sem"],
            yerr=row["one_minus_clisi_sem"],
            fmt="o",
            markersize=7.0 if method.startswith("CAMP") else 6.0,
            color=PUBLICATION_PALETTE.get(method, "#555555"),
            ecolor=PUBLICATION_PALETTE.get(method, "#555555"),
            elinewidth=1.0,
            capsize=2.0,
            alpha=0.95,
        )
    ax.set_xlabel("Donor mixing (iLISI; higher is better)")
    ax.set_ylabel("Cell-type conservation (1-cLISI; higher is better)")
    ax.grid(True, color="#d9d9d9", linewidth=0.6, alpha=0.65)
    sns.despine(ax=ax)
    fig.tight_layout()
    save_publication_figure(
        fig,
        supplement_dir / "figS07_batch_tradeoff_m1000_all_methods.png",
        args.dpi,
    )
    for metric, ylabel, filename in [
        (
            "iLISI_normalized",
            "Donor mixing (iLISI; higher is better)",
            "figS08_batch_ilisi_m1000_all_methods.png",
        ),
        (
            "one_minus_cLISI_normalized",
            "Cell-type conservation (1-cLISI; higher is better)",
            "figS09_batch_one_minus_clisi_m1000_all_methods.png",
        ),
    ]:
        plot_publication_boxplot(
            lisi[lisi["metric"] == metric],
            "score",
            ylabel,
            supplement_dir / filename,
            args.dpi,
        )

    validation_path = (
        output_root
        / "generated_partitions"
        / "matched_count_grid"
        / "fig4_matched_count_validation.csv"
    )
    if validation_path.is_file():
        validation = pd.read_csv(validation_path)
        for _, row in validation.iterrows():
            audit_rows.append(
                {
                    "experiment": "batch_integration_and_b_lineage",
                    "setting": f"m{int(row['target_metacells'])}",
                    "method": publication_method_label(row["method"]),
                    "requested_profiles": int(row["target_metacells"]),
                    "realized_profiles": int(row["realized_metacells"]),
                    "exact_match": bool(row["exact_target_pass"]),
                }
            )

    # DE preservation. Main text uses the prespecified default CAMP1 plus every
    # competitor; CAMP2-4 are retained in the complete supplementary panels.
    fig5_root = (
        output_root
        / "intermediate_csv"
        / "fig5_de_preservation"
        / "matched_reduction_rates"
    )
    concordance_frames = []
    correlation_frames = []
    for rate in args.de_reduction_rates:
        rate_dir = fig5_root / reduction_rate_slug(float(rate))
        concordance = read_csvs(rate_dir.glob("*_celltype_rank_concordance.csv"))
        correlations = read_csvs(rate_dir.glob("*_condition_correlations.csv"))
        if not concordance.empty:
            concordance["compression"] = float(rate)
            concordance_frames.append(concordance)
        if not correlations.empty:
            correlations["compression"] = float(rate)
            correlation_frames.append(correlations)
        summary_path = rate_dir / "metacell_summary.csv"
        if summary_path.is_file():
            summary = pd.read_csv(summary_path)
            for _, row in summary.iterrows():
                audit_rows.append(
                    {
                        "experiment": "differential_expression",
                        "setting": f"{float(rate):g}x",
                        "method": publication_method_label(row["method"]),
                        "requested_profiles": int(row["shared_target_metacells"]),
                        "realized_profiles": int(row["n_metacells"]),
                        "exact_match": bool(row["all_stratum_targets_matched"]),
                    }
                )
    if not concordance_frames or not correlation_frames:
        raise FileNotFoundError("Matched DE result tables are incomplete")
    concordance_all = pd.concat(concordance_frames, ignore_index=True)
    correlations_all = pd.concat(correlation_frames, ignore_index=True)
    methods_for_legend.update(concordance_all["method"].map(publication_method_label))
    primary_rate = float(args.de_primary_reduction_rate)
    primary_rate_dir = fig5_root / reduction_rate_slug(primary_rate)
    condition_examples = read_condition_de_examples(primary_rate_dir)
    condition_example_color_limit = condition_de_example_color_limit(
        condition_examples
    )
    main_example_methods = ["Full cells", *DEFAULT_FOCUSED_METHOD_ORDER]
    all_example_methods = ["Full cells", *PUBLICATION_METHOD_ORDER]
    plot_condition_de_example_heatmap(
        condition_examples,
        main_example_methods,
        main_dir / "figB_de_condition_gene_examples_camp1.png",
        args.dpi,
        condition_example_color_limit,
    )
    plot_condition_de_example_heatmap(
        condition_examples,
        all_example_methods,
        supplement_dir / "figS01_de_condition_gene_examples_all_methods.png",
        args.dpi,
        condition_example_color_limit,
    )
    save_condition_de_colorbar(
        main_dir / "de_condition_logfoldchange_colorbar.png",
        args.dpi,
        condition_example_color_limit,
    )
    save_csv(
        condition_examples[condition_examples["method"].isin(main_example_methods)],
        figure_data_dir / "figB_de_condition_gene_examples_camp1.csv",
    )
    save_csv(
        condition_examples,
        figure_data_dir / "figS01_de_condition_gene_examples_all_methods.csv",
    )
    plot_publication_boxplot(
        concordance_all[np.isclose(concordance_all["compression"], primary_rate)],
        "kendall_tau",
        "Kendall tau with full-cell DE ranks",
        supplement_dir / "figS05_de_celltype_kendall_m10_all_methods.png",
        args.dpi,
    )
    plot_publication_boxplot(
        correlations_all[np.isclose(correlations_all["compression"], primary_rate)],
        "pearson_r",
        "Pearson correlation with full-cell log fold changes",
        supplement_dir / "figS06_de_condition_pearson_m10_all_methods.png",
        args.dpi,
    )
    plot_publication_sensitivity(
        concordance_all,
        "compression",
        "kendall_tau",
        "Cell-type DE rank consistency (Kendall tau)",
        supplement_dir / "figS04_de_celltype_sensitivity_all_methods.png",
        args.dpi,
        x_tick_suffix="x",
        minimum_y_span=0.10,
        y_bounds=(-1.0, 1.0),
    )
    plot_publication_sensitivity(
        correlations_all,
        "compression",
        "pearson_r",
        "Condition-DE consistency (Pearson correlation)",
        main_dir / "figC_de_condition_sensitivity.png",
        args.dpi,
        x_tick_suffix="x",
        method_subset=DEFAULT_FOCUSED_METHOD_ORDER,
        minimum_y_span=0.012,
        y_bounds=(0.0, 1.0),
    )
    plot_publication_sensitivity(
        correlations_all,
        "compression",
        "pearson_r",
        "Condition-DE consistency (Pearson correlation)",
        supplement_dir / "figS03_de_condition_sensitivity_all_methods.png",
        args.dpi,
        x_tick_suffix="x",
        minimum_y_span=0.025,
        y_bounds=(0.0, 1.0),
    )
    save_csv(
        publication_method_frame(correlations_all)[
            publication_method_frame(correlations_all)["method"].isin(
                DEFAULT_FOCUSED_METHOD_ORDER
            )
        ],
        figure_data_dir / "figC_de_condition_sensitivity_default_camp1.csv",
    )
    metaq_fig5_panel_dir = (
        output_root
        / "figures"
        / "metaq_paper_style_matched_counts"
        / "fig5"
        / "individual_panels"
    )
    copy_figure_pair(
        metaq_fig5_panel_dir / "fig5b_rank_heatmap_pair__camp1",
        main_dir / "figA_de_rank_heatmap_camp1",
        "main_text_de_rank_heatmap",
        method="CAMP1",
    )
    metaq_fig5_meta_dir = (
        output_root
        / "intermediate_csv"
        / "metaq_paper_style_matched_counts"
        / "fig5"
    )
    logfc_long = pd.read_csv(metaq_fig5_meta_dir / "fig5b_logfoldchanges.csv.gz")
    selected_celltypes = (
        pd.read_csv(metaq_fig5_meta_dir / "fig5_selected_celltypes.csv")
        .sort_values("selection_order")["celltype"]
        .astype(str)
        .tolist()
    )
    selected_genes = (
        pd.read_csv(metaq_fig5_meta_dir / "fig5_selected_genes.csv")
        .sort_values("selection_order")["gene"]
        .astype(str)
        .tolist()
    )
    rank_matrices: Dict[str, pd.DataFrame] = {}
    for method, method_frame in logfc_long.groupby("method", sort=False):
        matrix = method_frame.pivot(
            index="celltype", columns="gene", values="logfoldchanges"
        ).reindex(index=selected_celltypes, columns=selected_genes)
        rank_matrices[str(method)] = logfc_to_rank_matrix(matrix)
    all_method_heatmap_path = (
        supplement_dir / "figS02_de_rank_heatmap_grid_all_methods.png"
    )
    plot_metaq_rank_heatmap_grid(
        rank_matrices,
        ["Original cells"] + ALL_METHOD_ORDER,
        all_method_heatmap_path,
        args.dpi,
    )
    manifest_rows.append(
        {
            "category": "supplement_de_rank_heatmap_grid",
            "method": "",
            "marker": "",
            "pdf": str(all_method_heatmap_path.with_suffix(".pdf").relative_to(output_dir)),
            "png": str(all_method_heatmap_path.relative_to(output_dir)),
        }
    )

    # B-lineage composition and marker fidelity at the same exact global
    # metacell counts used by Fig. 4. Main text again shows only CAMP1 and the
    # competitors, while the all-variant views remain supplementary.
    blineage_root = output_root / "intermediate_csv" / str(args.blineage_output_subdir)
    blineage_protocol_path = blineage_root / "blineage_analysis_protocol.json"
    if not blineage_protocol_path.is_file():
        raise FileNotFoundError(
            "Run --stage blineage once to create the exact-count B-lineage tables: "
            f"{blineage_protocol_path}"
        )
    blineage_protocol = json.loads(blineage_protocol_path.read_text())
    if (
        blineage_protocol.get("protocol") != BLINEAGE_PROTOCOL_VERSION
        or not bool(blineage_protocol.get("exact_count_matched", False))
    ):
        raise ValueError(
            "The B-lineage tables are not from the exact-count protocol; rerun "
            "--stage blineage with the current script."
        )
    proportion = pd.read_csv(
        blineage_root / "blineage_proportion_preservation.csv"
    )
    markers = pd.read_csv(
        blineage_root / "blineage_marker_reconstruction_metrics.csv"
    )
    methods_for_legend.update(proportion["method"].map(publication_method_label))
    plot_publication_sensitivity(
        proportion,
        "requested_metacells",
        "total_variation_error",
        "B-lineage proportion error (lower is better)",
        main_dir / "figD_blineage_proportion_error.png",
        args.dpi,
        include_full_cells=True,
        method_subset=[*DEFAULT_FOCUSED_METHOD_ORDER, "Full cells"],
        minimum_y_span=0.12,
        y_bounds=(0.0, 1.0),
        anchor_zero=True,
    )
    plot_publication_sensitivity(
        proportion,
        "requested_metacells",
        "total_variation_error",
        "B-lineage proportion error (lower is better)",
        supplement_dir / "figS10_blineage_proportion_error_all_methods.png",
        args.dpi,
        include_full_cells=True,
        minimum_y_span=0.30,
        y_bounds=(0.0, 1.0),
        anchor_zero=True,
    )
    marker_by_donor = (
        markers.groupby(
            ["requested_metacells", "method", "donor"], as_index=False
        )["pearson"]
        .mean()
        .rename(columns={"pearson": "marker_pearson"})
    )
    plot_publication_sensitivity(
        marker_by_donor,
        "requested_metacells",
        "marker_pearson",
        "B-lineage marker reconstruction (Pearson correlation)",
        main_dir / "figE_blineage_marker_pearson.png",
        args.dpi,
        include_full_cells=True,
        method_subset=[*DEFAULT_FOCUSED_METHOD_ORDER, "Full cells"],
        minimum_y_span=0.68,
        y_bounds=(-1.0, 1.02),
    )
    plot_publication_sensitivity(
        marker_by_donor,
        "requested_metacells",
        "marker_pearson",
        "B-lineage marker reconstruction (Pearson correlation)",
        supplement_dir / "figS11_blineage_marker_pearson_all_methods.png",
        args.dpi,
        include_full_cells=True,
        minimum_y_span=0.68,
        y_bounds=(-1.0, 1.02),
    )
    proportion_display = publication_method_frame(proportion)
    marker_display = publication_method_frame(marker_by_donor)
    save_csv(
        proportion_display[
            proportion_display["method"].isin(
                [*DEFAULT_FOCUSED_METHOD_ORDER, "Full cells"]
            )
        ],
        figure_data_dir / "figD_blineage_proportion_error_default_camp1.csv",
    )
    save_csv(
        marker_display[
            marker_display["method"].isin(
                [*DEFAULT_FOCUSED_METHOD_ORDER, "Full cells"]
            )
        ],
        figure_data_dir / "figE_blineage_marker_pearson_default_camp1.csv",
    )

    donor_metrics = pd.read_csv(
        blineage_root / "blineage_donor_embedding_metrics.csv"
    )
    plot_publication_sensitivity(
        donor_metrics[
            donor_metrics["metric"] == "leave_one_donor_balanced_accuracy"
        ],
        "requested_metacells",
        "score",
        "Leave-one-donor B-state balanced accuracy",
        supplement_dir / "figS13_blineage_leave_one_donor_accuracy_all_methods.png",
        args.dpi,
        include_full_cells=True,
        minimum_y_span=0.26,
        y_bounds=(0.0, 1.0),
    )
    plot_publication_sensitivity(
        donor_metrics[
            donor_metrics["metric"] == "within_donor_neighbor_purity"
        ],
        "requested_metacells",
        "score",
        "Within-donor B-state neighborhood purity",
        supplement_dir / "figS12_blineage_neighbor_purity_all_methods.png",
        args.dpi,
        include_full_cells=True,
        minimum_y_span=0.34,
        y_bounds=(0.0, 1.0),
    )
    mixing = pd.read_csv(
        blineage_root / "blineage_within_state_donor_mixing.csv"
    )
    plot_publication_sensitivity(
        mixing,
        "requested_metacells",
        "donor_mixing_simpson_normalized",
        "Donor mixing within B-lineage states",
        supplement_dir / "figS14_blineage_donor_mixing_all_methods.png",
        args.dpi,
        include_full_cells=True,
        minimum_y_span=0.24,
        y_bounds=(0.0, 1.0),
    )

    # Redraw the complete all-method B-lineage atlas from the plot-ready CSV.
    # UMAP state labels and normalized-expression scales are stored as separate
    # horizontal, frameless legends so every panel remains compact.
    coordinate_path = (
        blineage_root
        / f"blineage_primary_m{args.fig4_primary_target}_marker_coordinates.csv.gz"
    )
    if not coordinate_path.is_file():
        raise FileNotFoundError(f"Missing B-lineage plotting coordinates: {coordinate_path}")
    coordinates = pd.read_csv(coordinate_path)
    primary_frames = {
        str(method): method_frame.copy()
        for method, method_frame in coordinates.groupby("method", sort=False)
    }
    blineage_celltypes = [str(value) for value in blineage_protocol["celltypes"]]
    blineage_donors = [str(value) for value in blineage_protocol["selected_donors"]]
    blineage_condition = str(blineage_protocol["condition"])
    for method in blineage_method_order(primary_frames):
        method_slug = slugify(method).lower()
        destination = blineage_umap_dir / f"blineage_subtypes__{method_slug}.png"
        plot_blineage_annotation_facets(
            primary_frames[method],
            method,
            blineage_celltypes,
            blineage_donors,
            blineage_condition,
            destination,
            args.dpi,
        )
        manifest_rows.append(
            {
                "category": "supplement_blineage_umap",
                "method": publication_method_label(method),
                "marker": "",
                "pdf": str(destination.with_suffix(".pdf").relative_to(output_dir)),
                "png": str(destination.relative_to(output_dir)),
            }
        )
    state_legend_path = standalone_legend_dir / "blineage_state_legend.png"
    save_blineage_state_legend(
        blineage_celltypes,
        state_legend_path,
        args.dpi,
    )
    marker_genes = [str(gene) for gene in blineage_protocol["marker_genes_plotted"]]
    for gene in marker_genes:
        shared_vmax = max(1e-6, float(coordinates[gene].quantile(0.99)))
        for method in blineage_method_order(primary_frames):
            if method == "Harmony":
                continue
            method_slug = slugify(method).lower()
            marker_slug = slugify(gene).lower()
            destination = (
                blineage_marker_dir
                / f"blineage_{marker_slug}__{method_slug}_vs_full_cells.png"
            )
            plot_blineage_marker_method_vs_harmony(
                primary_frames,
                method,
                gene,
                blineage_donors,
                blineage_condition,
                destination,
                args.dpi,
                vmax_override=shared_vmax,
            )
            manifest_rows.append(
                {
                    "category": "supplement_blineage_marker_map",
                    "method": publication_method_label(method),
                    "marker": gene,
                    "pdf": str(destination.with_suffix(".pdf").relative_to(output_dir)),
                    "png": str(destination.relative_to(output_dir)),
                }
            )
        expression_legend_path = (
            standalone_legend_dir
            / f"blineage_{slugify(gene).lower()}_normalized_expression_legend.png"
        )
        save_horizontal_expression_legend(
            gene,
            shared_vmax,
            expression_legend_path,
            args.dpi,
        )

    save_focused_method_legend(
        [*DEFAULT_FOCUSED_METHOD_ORDER, "Full cells"],
        main_dir / "shared_method_legend_default_camp1.png",
        args.dpi,
    )
    save_focused_method_legend(
        sorted(methods_for_legend),
        supplement_dir / "figS15_shared_method_legend_all_methods.png",
        args.dpi,
    )
    listed_pdfs = {str(row["pdf"]) for row in manifest_rows}
    for pdf_path in sorted(output_dir.rglob("*.pdf")):
        relative_pdf = str(pdf_path.relative_to(output_dir))
        if relative_pdf in listed_pdfs:
            continue
        png_path = pdf_path.with_suffix(".png")
        manifest_rows.append(
            {
                "category": (
                    "main_text_panel"
                    if relative_pdf.startswith("main_text/")
                    else "supplement_panel"
                ),
                "method": "",
                "marker": "",
                "pdf": relative_pdf,
                "png": (
                    str(png_path.relative_to(output_dir))
                    if png_path.is_file()
                    else ""
                ),
            }
        )
    save_csv(
        pd.DataFrame(manifest_rows).sort_values(["category", "pdf"]),
        output_dir / "supplemental_file_manifest.csv",
    )
    audit = pd.DataFrame(audit_rows)
    save_csv(audit, output_dir / "compression_matching_audit.csv")
    if not audit.empty and not bool(audit["exact_match"].all()):
        raise RuntimeError("At least one displayed method missed its matched profile budget")
    save_json(
        {
            "protocol": "default_camp1_manuscript_panels_matched_v4",
            "main_story": [
                "representative COVID-versus-Healthy differential-expression effects",
                "default CAMP1 condition differential-expression preservation",
                "default CAMP1 B-lineage composition preservation",
                "default CAMP1 B-lineage marker-expression reconstruction",
            ],
            "main_text_method_policy": (
                "Prespecified default CAMP1 versus all five competing methods; "
                "CAMP2-4 appear only in the complete all-method supplement."
            ),
            "default_focused_methods": DEFAULT_FOCUSED_METHOD_ORDER,
            "fig4_metacell_counts": args.fig4_target_metacells,
            "fig4_primary_metacells": args.fig4_primary_target,
            "de_compression_rates": args.de_reduction_rates,
            "de_primary_compression": args.de_primary_reduction_rate,
            "blineage_protocol": BLINEAGE_PROTOCOL_VERSION,
            "method_palette": PUBLICATION_PALETTE,
            "main_text_panels": [
                "figA_de_rank_heatmap_camp1",
                "figB_de_condition_gene_examples_camp1",
                "figC_de_condition_sensitivity",
                "figD_blineage_proportion_error",
                "figE_blineage_marker_pearson",
            ],
            "supplement_contains": [
                "all-method representative condition-DE gene effects",
                "all-method DE heatmap grid",
                "all-method condition- and cell-type-DE sensitivity",
                "all-method matched-10x DE boxplots",
                "batch trade-off, iLISI, and 1-cLISI",
                "all-method B-lineage proportion and marker reconstruction",
                "B-state neighborhood purity, cross-donor recovery, and donor mixing",
                "all-method B-lineage UMAPs and five-gene marker maps",
                "frameless horizontal method, B-state, and marker-expression legends",
            ],
        },
        output_dir / "figure_set_protocol.json",
    )
    logger.info("Focused manuscript panels saved: %s", output_dir)


# =========================================================
# Driver / CLI
# =========================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "CAMP batch-integration and DE-preservation experiments with "
            "dataset-specific released native memberships"
        ),
    )
    parser.add_argument(
        "--dataset-name",
        choices=["pbmc", "human_fetal_atlas"],
        default="pbmc",
        help="Selects preprocessing, released memberships, cache names, and labels",
    )
    parser.add_argument(
        "--stage",
        choices=[
            "all",
            "compute",
            "fig4",
            "de",
            "native_de",
            "matched",
            "plot",
            "metaq_plot",
            "blineage",
            "manuscript_plot",
            "inspect",
        ],
        default="all",
        help=(
            "Use 'matched' for the complete count-controlled Fig. 4/5 analysis, "
            "'fig4' for matched-count batch integration only, or 'de' for "
            "matched within-stratum DE preservation only. Use 'native_de' to "
            "evaluate the original native partitions nearest the requested global "
            "metacell counts, intersected only with cell type, donor, and condition "
            "boundaries. Use 'metaq_plot' to "
            "render count-controlled MetaQ Fig. 4/5-style panels. Use "
            "'blineage' for the exact-count donor-blocked B-lineage analysis, "
            "or 'manuscript_plot' to redraw the final DE-only paper panels "
            "from completed result tables without a B-lineage dependency."
        ),
    )
    parser.add_argument("--input-h5ad", type=str, default="/storage/home/dvl5760/scratch/blish_covid.seu.h5ad")
    parser.add_argument("--output-dir", type=str, default="./results_pbmc_native_fig4_fig5_v2")
    parser.add_argument("--fig4-variants", nargs="+", default=["camp1", "camp2", "camp3", "camp4"])
    parser.add_argument(
        "--fig4-gamma",
        type=int,
        default=20,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--fig4-target-metacells",
        nargs="+",
        type=int,
        default=[750, 1000, 1250],
        help=(
            "Exact shared Fig. 4 metacell counts. Each method starts from a native "
            "high-resolution partition and uses the same merge-only coarsening rule."
        ),
    )
    parser.add_argument(
        "--fig4-primary-target",
        type=int,
        default=1000,
        help="Requested grid point used for the primary original-cell reference",
    )
    parser.add_argument(
        "--baseline-root",
        type=str,
        default=str(LOCAL_SEACELLS_SOURCE),
        help="Root of the uploaded SEACells bundle containing released memberships",
    )
    parser.add_argument(
        "--skip-baselines",
        action="store_true",
        help="Run only CAMP1-4 (normally leave this disabled)",
    )

    parser.add_argument("--celltype-key", type=str, default="auto")
    parser.add_argument("--batch-key", type=str, default="auto")
    parser.add_argument("--donor-key", type=str, default="auto")
    parser.add_argument("--condition-key", type=str, default="auto")
    parser.add_argument(
        "--condition-display-name",
        type=str,
        default="Condition",
        help="Human-readable condition name used in Fig. 5 labels (for HFA: Organ)",
    )
    parser.add_argument("--reference-condition", type=str, default="auto")
    parser.add_argument("--loom-var-name-key", type=str, default="gene_short_name")

    parser.add_argument("--min-genes", type=int, default=200)
    parser.add_argument("--min-cells", type=int, default=3)
    parser.add_argument("--normalize-target-sum", type=float, default=1e4)
    parser.add_argument("--n-hvg", type=int, default=2000)
    parser.add_argument("--n-pcs", type=int, default=50)
    parser.add_argument("--n-neighbors", type=int, default=15)
    parser.add_argument("--harmony-iterations", type=int, default=30)
    parser.add_argument("--mapping-epochs", type=int, default=1000)
    parser.add_argument("--mapping-batch-size", type=int, default=512)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--recovered-resolutions", nargs="+", type=float, default=[0.5, 1.0, 2.0])
    parser.add_argument("--metacell-resolutions", nargs="+", type=float, default=[1.0, 2.0, 5.0])
    parser.add_argument("--lisi-perplexity", type=int, default=30)

    parser.add_argument("--de-top-genes", type=int, default=2000)
    parser.add_argument(
        "--de-reduction-rates",
        nargs="+",
        type=float,
        default=[8.0, 10.0, 12.0],
        help=(
            "Exact shared within-stratum compression rates for Fig. 5; 10x "
            "matches the MetaQ differential-expression protocol."
        ),
    )
    parser.add_argument(
        "--de-primary-reduction-rate",
        type=float,
        default=10.0,
        help="Matched DE rate used for the primary MetaQ-style Fig. 5 panels",
    )

    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument(
        "--metaq-marker-gene",
        type=str,
        default="MS4A1",
        help="Marker gene used in the MetaQ Fig. 4f-style panel",
    )
    parser.add_argument(
        "--metaq-marker-celltype",
        type=str,
        default="B",
        help="Cell type used in Fig. 4f; 'auto' selects the highest-mean marker cell type",
    )
    parser.add_argument(
        "--metaq-focus-celltype",
        type=str,
        default="CD8eff T",
        help="Cell type used in the MetaQ Fig. 5d-style condition-DE heatmaps",
    )
    parser.add_argument(
        "--metaq-top-celltypes",
        type=int,
        default=6,
        help="Number of abundant reference-condition cell types in Fig. 5b/c",
    )
    parser.add_argument(
        "--metaq-top-genes-per-celltype",
        type=int,
        default=10,
        help="Original-data marker genes contributed by each Fig. 5b cell type",
    )
    parser.add_argument(
        "--blineage-output-subdir",
        type=str,
        default="b_lineage_heterogeneity_matched_counts_v2",
        help="New figures/intermediate subfolder; existing Fig. 4/5 outputs are untouched",
    )
    parser.add_argument(
        "--blineage-condition",
        type=str,
        default="COVID",
        help="Condition used for donor-blocked B-lineage validation; use 'auto' to infer it",
    )
    parser.add_argument(
        "--blineage-celltypes",
        nargs="+",
        default=["B", "Class-switched B", "IgA PB", "IgG PB"],
        help="Known B-lineage states evaluated together",
    )
    parser.add_argument(
        "--blineage-donors",
        nargs="+",
        default=["auto"],
        help="Biological donors to validate; 'auto' keeps donors meeting the per-state minimum",
    )
    parser.add_argument(
        "--blineage-min-cells-per-state",
        type=int,
        default=20,
        help="Minimum cells from every requested B-lineage state in each selected donor",
    )
    parser.add_argument(
        "--blineage-neighbors",
        type=int,
        default=15,
        help="Neighbors used for donor-blocked subtype fidelity and donor-mixing metrics",
    )
    parser.add_argument(
        "--blineage-marker-genes",
        nargs="+",
        default=[
            "MS4A1",
            "IGHD",
            "TCL1A",
            "MZB1",
            "XBP1",
            "PRDM1",
            "CD38",
            "IGHA1",
            "IGHG1",
        ],
        help="Markers included in metacell-expression reconstruction metrics",
    )
    parser.add_argument(
        "--blineage-plot-genes",
        nargs="+",
        default=["MS4A1", "IGHD", "MZB1", "IGHA1", "IGHG1"],
        help="Curated markers saved as individual MetaQ-style donor-faceted plots",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--force-fig4",
        action="store_true",
        help="Recompute completed exact-count Fig. 4 targets",
    )
    parser.add_argument(
        "--force-de",
        action="store_true",
        help="Replace only matched Fig. 5 outputs; preprocessing and Fig. 4 stay cached",
    )
    return parser.parse_args()


def main() -> None:
    global RELEASED_BASELINES
    args = parse_args()
    dataset_slug = slugify(args.dataset_name).lower()
    if dataset_slug == "human_fetal_atlas":
        RELEASED_BASELINES = RELEASED_HFA_BASELINES
    elif dataset_slug == "pbmc":
        RELEASED_BASELINES = RELEASED_PBMC_BASELINES
    else:
        raise ValueError(
            "--dataset-name must be 'pbmc' or 'human_fetal_atlas' for the "
            "released-membership comparison"
        )
    if dataset_slug != "pbmc" and args.stage in {"blineage", "manuscript_plot"}:
        raise ValueError(
            f"--stage {args.stage} is PBMC-specific; use a Fig. 4/5 stage for "
            "human_fetal_atlas"
        )
    args.fig4_target_metacells = [int(value) for value in args.fig4_target_metacells]
    if len(set(args.fig4_target_metacells)) != len(args.fig4_target_metacells):
        raise ValueError("--fig4-target-metacells contains duplicate values")
    if args.fig4_primary_target not in args.fig4_target_metacells:
        raise ValueError(
            "--fig4-primary-target must be one of --fig4-target-metacells"
        )
    args.de_reduction_rates = [float(value) for value in args.de_reduction_rates]
    if any(value <= 1 for value in args.de_reduction_rates):
        raise ValueError("--de-reduction-rates values must all be greater than 1")
    if len(set(args.de_reduction_rates)) != len(args.de_reduction_rates):
        raise ValueError("--de-reduction-rates contains duplicate values")
    if not any(
        np.isclose(args.de_primary_reduction_rate, value)
        for value in args.de_reduction_rates
    ):
        raise ValueError(
            "--de-primary-reduction-rate must be one of --de-reduction-rates"
        )
    if len(set(args.blineage_celltypes)) != len(args.blineage_celltypes):
        raise ValueError("--blineage-celltypes contains duplicate values")
    if args.blineage_min_cells_per_state < 2:
        raise ValueError("--blineage-min-cells-per-state must be at least 2")
    if args.blineage_neighbors < 1:
        raise ValueError("--blineage-neighbors must be positive")
    blineage_subdir = Path(args.blineage_output_subdir)
    if (
        blineage_subdir.is_absolute()
        or len(blineage_subdir.parts) != 1
        or args.blineage_output_subdir in {"", ".", ".."}
    ):
        raise ValueError("--blineage-output-subdir must be one safe folder name")
    set_random_seed(args.random_seed)
    sns.set_theme(style="ticks", context="paper", font_scale=1.1)

    output_root = Path(args.output_dir).expanduser().resolve()
    cache_dir = output_root / "cache"
    fig4_native_intermediate = (
        output_root / "intermediate_csv" / "fig4_batch_integration" / "native_resolutions"
    )
    fig4_matched_intermediate = (
        output_root / "intermediate_csv" / "fig4_batch_integration" / "matched_counts"
    )
    fig5_native_intermediate = (
        output_root / "intermediate_csv" / "fig5_de_preservation" / "native_resolutions"
    )
    fig5_matched_intermediate = (
        output_root
        / "intermediate_csv"
        / "fig5_de_preservation"
        / "matched_reduction_rates"
    )
    generated_partition_dir = output_root / "generated_partitions"
    figure_dir = output_root / "figures"
    for path in [
        cache_dir,
        fig4_native_intermediate,
        fig4_matched_intermediate,
        fig5_native_intermediate,
        fig5_matched_intermediate,
        generated_partition_dir,
        figure_dir,
    ]:
        safe_mkdir(path)

    run_start_path = output_root / "run_start_config.json"
    if args.stage == "blineage":
        run_start_path = (
            output_root
            / "intermediate_csv"
            / str(args.blineage_output_subdir)
            / "blineage_run_start_config.json"
        )
    save_json(
        {
            "started_at": pd.Timestamp.now().isoformat(),
            "dataset_name": args.dataset_name,
            "script": str(Path(__file__).resolve()),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "stage": args.stage,
            "output_root": str(output_root),
            "fig4_protocol": MATCHED_FIG4_PROTOCOL_VERSION,
            "de_protocol": MATCHED_DE_PROTOCOL_VERSION,
            "requested_metacells": args.fig4_target_metacells,
            "primary_requested_metacells": args.fig4_primary_target,
            "de_reduction_rates": args.de_reduction_rates,
            "de_primary_reduction_rate": args.de_primary_reduction_rate,
            "posthoc_kmeans_used_for_count_control": args.stage != "native_de",
        },
        run_start_path,
    )

    if args.stage in {
        "all",
        "compute",
        "fig4",
        "de",
        "native_de",
        "matched",
        "inspect",
    }:
        base, model = build_or_load_preprocessing_cache(args, cache_dir)
        celltype_key = resolve_obs_key(
            base.obs, args.celltype_key, CELLTYPE_CANDIDATES, "celltype"
        )
        batch_key = resolve_obs_key(base.obs, args.batch_key, BATCH_CANDIDATES, "batch")
        donor_key = resolve_obs_key(
            base.obs, args.donor_key, DONOR_CANDIDATES, "donor", required=False
        )
        if donor_key is None:
            donor_key = batch_key
            logger.warning("No donor key found; using batch key '%s' as donor", batch_key)
        condition_key = resolve_obs_key(
            base.obs, args.condition_key, CONDITION_CANDIDATES, "condition"
        )
        keys = {
            "celltype": celltype_key,
            "batch": batch_key,
            "donor": donor_key,
            "condition": condition_key,
        }
        reference_condition = choose_reference_condition(
            base.obs[condition_key], args.reference_condition
        )
        key_summary = pd.DataFrame(
            [
                {
                    "celltype_key": celltype_key,
                    "batch_key": batch_key,
                    "donor_key": donor_key,
                    "condition_key": condition_key,
                    "reference_condition": reference_condition,
                    "n_celltypes": clean_obs_values(base.obs[celltype_key]).nunique(),
                    "n_batches": clean_obs_values(base.obs[batch_key]).nunique(),
                    "n_donors": clean_obs_values(base.obs[donor_key]).nunique(),
                    "n_conditions": clean_obs_values(base.obs[condition_key]).nunique(),
                }
            ]
        )
        save_csv(key_summary, output_root / "resolved_metadata_keys.csv")
        logger.info("Resolved metadata:\n%s", key_summary.to_string(index=False))

        if dataset_slug == "human_fetal_atlas":
            availability_rows = [
                {
                    "method": method,
                    "included": True,
                    "reason": "Full-atlas released native membership is available.",
                }
                for method in RELEASED_HFA_BASELINES
            ]
            availability_rows.extend(
                {
                    "method": method,
                    "included": False,
                    "reason": reason,
                }
                for method, reason in HFA_UNAVAILABLE_BASELINES.items()
            )
            save_csv(
                pd.DataFrame(availability_rows),
                output_root / "hfa_method_availability.csv",
            )

        if args.stage == "inspect":
            logger.info("Inspection requested; stopping before experiments")
            return

        if not args.skip_baselines:
            baseline_grid, evaluation_ids, baseline_summary = load_released_pbmc_native_grid(
                model=model,
                baseline_root=Path(args.baseline_root).expanduser().resolve(),
                targets=args.fig4_target_metacells,
                partition_dir=generated_partition_dir,
            )
            base_evaluation = base[evaluation_ids].copy()
            model_evaluation = model[evaluation_ids].copy()
        else:
            evaluation_ids = model.obs_names.copy()
            base_evaluation = base
            model_evaluation = model
            baseline_grid = {
                int(target): {} for target in args.fig4_target_metacells
            }
            baseline_summary = pd.DataFrame()

        if dataset_slug == "human_fetal_atlas" and not args.skip_baselines:
            # The released CAMP HFA memberships were generated by the GitHub
            # on-the-fly pipeline.  Reuse them as native method output and apply
            # exactly the same count-matching rule as every comparator.
            camp_grid = {
                int(target): {} for target in args.fig4_target_metacells
            }
            camp_summary = pd.DataFrame()
            logger.info(
                "Using released on-the-fly HFA memberships for CAMP1-4; "
                "no PBMC-style CAMP partition is recomputed."
            )
        else:
            camp_grid, camp_summary = build_or_load_camp_native_grid(
                model=model_evaluation,
                variants=args.fig4_variants,
                targets=args.fig4_target_metacells,
                seed=args.random_seed,
                partition_dir=generated_partition_dir,
                force=args.force or args.force_fig4,
            )
        assignments_by_target: Dict[int, Dict[str, pd.Series]] = {}
        for target in args.fig4_target_metacells:
            assignments_by_target[target] = dict(camp_grid[target])
            assignments_by_target[target].update(baseline_grid[target])
        comparison_methods = list(assignments_by_target[args.fig4_primary_target])
        logger.info("Comparison methods: %s", ", ".join(comparison_methods))

        native_summary = pd.concat(
            [frame for frame in [camp_summary, baseline_summary] if not frame.empty],
            ignore_index=True,
        )
        save_csv(
            native_summary,
            generated_partition_dir / "native_resolution_grid" / "all_method_resolution_summary.csv",
        )

        # Native-resolution DE used by the manuscript's additional experiment.
        # This keeps each method's released/global partition nearest the requested
        # metacell count and only cuts groups at cell type x donor x condition
        # boundaries.  No post-hoc count matching or K-means is performed.
        if args.stage in {"all", "native_de"}:
            run_fig5_native_grid(
                args,
                base_evaluation,
                model_evaluation,
                keys,
                assignments_by_target,
                reference_condition,
                fig5_native_intermediate,
            )

        # Keep the former closest-native Fig. 4 outputs only as an audit when
        # running the full pipeline. Primary paper figures use exact counts.
        if args.stage == "all":
            for target in args.fig4_target_metacells:
                target_out_dir = fig4_native_intermediate / f"requested_m{target}"
                include_original = target == args.fig4_primary_target
                if not args.force_fig4 and fig4_target_is_complete(
                    target_out_dir,
                    list(assignments_by_target[target]),
                    include_original,
                    requested_metacells=target,
                ):
                    logger.info(
                        "Fig. 4 requested m=%d is complete; keeping checkpoints. "
                        "Use --force-fig4 to recompute it.",
                        target,
                    )
                    continue
                run_fig4(
                    args,
                    base_evaluation,
                    model_evaluation,
                    keys,
                    assignments_by_target[target],
                    target_out_dir,
                    target_metacells=target,
                    include_original=include_original,
                    comparison_mode="native",
                )

        matched_fig4_assignments: Dict[int, Dict[str, pd.Series]] = {}
        if args.stage in {"all", "compute", "fig4", "matched"}:
            matched_sources_by_target: Dict[int, Dict[str, pd.Series]] = {
                target: dict(camp_grid[target])
                for target in args.fig4_target_metacells
            }
            if not args.skip_baselines:
                baseline_sources_by_target, _ = load_released_pbmc_merge_sources(
                    model=model_evaluation,
                    baseline_root=Path(args.baseline_root).expanduser().resolve(),
                    common_ids=evaluation_ids,
                    targets=args.fig4_target_metacells,
                    partition_dir=generated_partition_dir,
                )
                for target in args.fig4_target_metacells:
                    matched_sources_by_target[target].update(
                        baseline_sources_by_target[target]
                    )
            matched_fig4_assignments = build_matched_fig4_assignments(
                model=model_evaluation,
                source_assignments_by_target=matched_sources_by_target,
                targets=args.fig4_target_metacells,
                seed=args.random_seed,
                output_dir=generated_partition_dir / "matched_count_grid",
            )
            for target in args.fig4_target_metacells:
                target_out_dir = fig4_matched_intermediate / f"m{target}"
                include_original = target == args.fig4_primary_target
                if not args.force_fig4 and fig4_target_is_complete(
                    target_out_dir,
                    list(matched_fig4_assignments[target]),
                    include_original,
                    requested_metacells=target,
                    expected_protocol=MATCHED_FIG4_PROTOCOL_VERSION,
                    expected_posthoc_kmeans=True,
                ):
                    logger.info(
                        "Matched Fig. 4 m=%d is complete; keeping checkpoints. "
                        "Use --force-fig4 to recompute it.",
                        target,
                    )
                    continue
                run_fig4(
                    args,
                    base_evaluation,
                    model_evaluation,
                    keys,
                    matched_fig4_assignments[target],
                    target_out_dir,
                    target_metacells=target,
                    include_original=include_original,
                    comparison_mode="matched_count",
                )

        if args.stage in {"all", "compute", "de", "matched"}:
            run_fig5_matched_grid(
                args,
                base_evaluation,
                model_evaluation,
                keys,
                assignments_by_target[args.fig4_primary_target],
                reference_condition,
                fig5_matched_intermediate,
            )

        manifest = vars(args).copy()
        manifest.update(
            {
                "resolved_keys": keys,
                "resolved_reference_condition": reference_condition,
                "comparison_methods": comparison_methods,
                "full_preprocessed_cells": model.n_obs,
                "evaluation_cells": len(evaluation_ids),
                "evaluation_cell_protocol": (
                    "native_membership_intersection"
                    if not args.skip_baselines
                    else "all_preprocessed_cells"
                ),
                "de_protocol": (
                    NATIVE_DE_PROTOCOL_VERSION
                    if args.stage == "native_de"
                    else MATCHED_DE_PROTOCOL_VERSION
                ),
                "fig4_protocol": MATCHED_FIG4_PROTOCOL_VERSION,
                "fig4_comparison_contract": (
                    "same cells and exact requested metacell counts for every method; "
                    "closest merge-feasible released native source at each target; "
                    "merge intact fragments with the same size-weighted PCA-centroid "
                    "K-means rule"
                ),
                "de_comparison_contract": (
                    "each method's native global partition nearest the requested "
                    "metacell count; intersect native groups with celltype x donor x "
                    "condition boundaries without merging or reclustering"
                    if args.stage == "native_de"
                    else "same primary global-resolution inputs; exact identical "
                    "metacell targets in every celltype x donor x condition stratum "
                    "at each requested compression rate; adjustment never crosses "
                    "a stratum"
                ),
                "posthoc_kmeans_used_for_count_control": args.stage != "native_de",
                "native_outputs_retained_as_secondary_audit": True,
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "completed_at": pd.Timestamp.now().isoformat(),
            }
        )
        manifest_stem = (
            "native_de_run_manifest"
            if args.stage == "native_de"
            else "matched_de_run_manifest"
            if args.stage == "de"
            else "matched_fig4_run_manifest"
            if args.stage == "fig4"
            else "matched_run_manifest"
            if args.stage == "matched"
            else "run_manifest"
        )
        save_json(manifest, output_root / f"{manifest_stem}.json")
        save_csv(
            pd.DataFrame(
                [{"parameter": key, "value": json.dumps(value, default=str)} for key, value in manifest.items()]
            ),
            output_root / f"{manifest_stem}.csv",
        )

    if args.stage in {"all", "plot", "fig4", "matched"}:
        plot_matched_fig4(
            fig4_matched_intermediate,
            figure_dir / "fig4_batch_integration" / "matched_counts",
            args.fig4_target_metacells,
            args.fig4_primary_target,
            args.dpi,
            dataset_display_name=(
                "Human Fetal Atlas"
                if dataset_slug == "human_fetal_atlas"
                else "PBMC"
            ),
        )
    if args.stage in {"all", "plot", "de", "matched"}:
        plot_matched_fig5_grid(
            fig5_matched_intermediate,
            figure_dir / "fig5_de_preservation" / "matched_reduction_rates",
            args.de_reduction_rates,
            args.de_primary_reduction_rate,
            args.dpi,
        )

    if args.stage in {"all", "plot", "metaq_plot", "matched"}:
        plot_metaq_paper_style_outputs(
            output_root=output_root,
            dataset_name=args.dataset_name,
            condition_display_name=args.condition_display_name,
            fig4_intermediate_root=fig4_matched_intermediate,
            fig5_intermediate_root=fig5_matched_intermediate,
            primary_target=args.fig4_primary_target,
            camp_variants=args.fig4_variants,
            marker_gene=args.metaq_marker_gene,
            marker_celltype=args.metaq_marker_celltype,
            focus_celltype=args.metaq_focus_celltype,
            top_celltypes=args.metaq_top_celltypes,
            top_genes_per_celltype=args.metaq_top_genes_per_celltype,
            dpi=args.dpi,
            fig4_source_override=(
                fig4_matched_intermediate / f"m{args.fig4_primary_target}"
            ),
            fig5_source_override=(
                fig5_matched_intermediate
                / reduction_rate_slug(args.de_primary_reduction_rate)
            ),
            output_subdir="metaq_paper_style_matched_counts",
        )

    if dataset_slug == "pbmc" and args.stage in {"all", "plot", "matched", "blineage"}:
        run_blineage_analysis(args, output_root)

    if dataset_slug == "pbmc" and args.stage in {
        "all",
        "plot",
        "matched",
        "blineage",
    }:
        plot_focused_manuscript_outputs(args, output_root)

    if dataset_slug == "pbmc" and args.stage == "manuscript_plot":
        plot_de_only_manuscript_outputs(args, output_root)

    logger.info("All requested work completed. Results: %s", output_root)


if __name__ == "__main__":
    main()
