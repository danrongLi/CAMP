#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import json
import argparse
import logging
from pathlib import Path
from typing import Optional, List, Dict, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("MKL_DYNAMIC", "FALSE")

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D
from anndata import AnnData
from scipy import sparse
from scipy.sparse import csr_matrix
from scipy.stats import entropy as scipy_entropy
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize

import SEACells
from SEACells import build_graph


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

# Hardcoded processed file for the special RECOMB CAMP4 setting
CAMP4_DIRECT_PROCESSED_FILE = "/storage/home/dvl5760/scratch/blish_covid_processed.h5ad"

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

VARIANT_ORDER = ["camp1", "camp2", "camp3", "camp4"]

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


# =========================================================
# Utilities
# =========================================================
def safe_mkdir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def clip_negative_sparse_inplace(X):
    if sparse.issparse(X):
        if X.nnz > 0:
            X.data = np.maximum(X.data, 0.0)
    else:
        X[:] = np.maximum(X, 0.0)


def get_total_counts(X):
    if sparse.issparse(X):
        return np.asarray(X.sum(axis=1)).ravel()
    return np.asarray(X.sum(axis=1)).ravel()


def compute_adaptive_kernel(
    X: np.ndarray,
    n_neighbors: int = 15,
    symmetrize: bool = True,
) -> csr_matrix:
    """
    Build a k-NN distance graph and convert it to an adaptive Gaussian kernel.
    This matches the RECOMB CAMP3 style.
    """
    from sklearn.neighbors import NearestNeighbors

    logger.info("Computing adaptive Gaussian kernel | n_neighbors=%d", n_neighbors)

    nbrs = NearestNeighbors(n_neighbors=n_neighbors)
    nbrs.fit(X)
    distances, indices = nbrs.kneighbors(X)

    n_cells = X.shape[0]
    rows = np.repeat(np.arange(n_cells), n_neighbors)
    cols = indices.flatten()
    data = distances.flatten()

    D = csr_matrix((data, (rows, cols)), shape=(n_cells, n_cells))

    if symmetrize:
        D = D.maximum(D.T)

    sigma = distances[:, -1]
    sigma = np.maximum(sigma, 1e-8)

    D_coo = D.tocoo()
    sim = np.exp(-(D_coo.data ** 2) / (sigma[D_coo.row] * sigma[D_coo.col]))

    K = csr_matrix((sim, (D_coo.row, D_coo.col)), shape=D.shape)
    logger.info("Adaptive kernel constructed | shape=%s | nnz=%d", K.shape, K.nnz)
    return K


def ensure_gene_names(adata: AnnData, loom_var_name_key: str = "gene_short_name") -> AnnData:
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


def read_dataset(input_file: str, loom_var_name_key: str = "gene_short_name") -> AnnData:
    ext = Path(input_file).suffix.lower()
    logger.info("Reading dataset: %s", input_file)

    if ext == ".h5ad":
        adata = sc.read_h5ad(input_file)
    elif ext == ".loom":
        adata = sc.read_loom(input_file, sparse=True, dtype="float32")
    else:
        raise ValueError(f"Unsupported input format: {input_file}")

    adata = ensure_gene_names(adata, loom_var_name_key=loom_var_name_key)
    return adata


def load_recomb_processed_direct(
    input_file: str,
    loom_var_name_key: str,
    n_pcs: int,
) -> AnnData:
    """
    Direct RECOMB-style loader for the already-processed file.

    Important:
    - no filtering
    - no normalize_total
    - no log1p
    - no HVG selection
    - no scaling
    - use existing X_pca if available; otherwise compute PCA with zero_center=False
    """
    adata = read_dataset(input_file, loom_var_name_key=loom_var_name_key)
    logger.info(
        "Using direct processed input: %s | skipping cache + skipping normalize/log1p/HVG/scale",
        input_file,
    )

    if "X_pca" in adata.obsm and adata.obsm["X_pca"].shape[1] >= n_pcs:
        adata.obsm["X_pca"] = np.asarray(adata.obsm["X_pca"][:, :n_pcs], dtype=np.float32)
        logger.info("Using existing X_pca from file | shape=%s", adata.obsm["X_pca"].shape)
    else:
        logger.info("X_pca missing (or too small); recomputing PCA with zero_center=False exactly as RECOMB CAMP4")
        sc.tl.pca(
            adata,
            n_comps=n_pcs,
            svd_solver="arpack",
            zero_center=False,
        )
        adata.obsm["X_pca"] = np.asarray(adata.obsm["X_pca"][:, :n_pcs], dtype=np.float32)
        logger.info("Computed X_pca | shape=%s", adata.obsm["X_pca"].shape)

    sc.pp.neighbors(adata, use_rep="X_pca")
    sc.tl.diffmap(adata)

    # CAMP2 RECOMB-style cosine similarity
    X_unit = normalize(adata.obsm["X_pca"], axis=1, copy=True)
    adata.obsm["X_unit"] = X_unit
    #adata.obsm["X_simi_cos"] = X_unit.dot(X_unit.T)

    # CAMP3 RECOMB-style adaptive Gaussian kernel
    adata.obsm["X_simi_adg"] = compute_adaptive_kernel(
        adata.obsm["X_pca"],
        n_neighbors=15,
        symmetrize=True,
    )

    logger.info(
        "Direct processed load finished | n_obs=%d | n_vars=%d | X_pca=%s",
        adata.n_obs,
        adata.n_vars,
        adata.obsm["X_pca"].shape,
    )
    return adata


def write_metadata_json(path: str, obj: Dict):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


# =========================================================
# Cached preprocessing
# =========================================================
def build_or_load_base_cache(
    input_file: str,
    dataset_name: str,
    cache_dir: str,
    loom_var_name_key: str,
    min_genes: int,
    min_cells: int,
    normalize_target_sum: float = 1e4,
) -> str:
    safe_mkdir(cache_dir)
    base_h5ad = os.path.join(cache_dir, f"{dataset_name}_BASE_log1p.h5ad")
    meta_json = os.path.join(cache_dir, f"{dataset_name}_BASE_log1p.meta.json")

    if os.path.exists(base_h5ad):
        logger.info("Using existing base cache: %s", base_h5ad)
        return base_h5ad

    logger.info("Building base cache for %s", dataset_name)
    adata = read_dataset(input_file, loom_var_name_key=loom_var_name_key)

    logger.info("Raw shape: %s", adata.shape)
    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_genes(adata, min_cells=min_cells)
    logger.info("After filtering: %s", adata.shape)

    clip_negative_sparse_inplace(adata.X)
    total_counts = get_total_counts(adata.X)
    logger.info(
        "Counts before normalization: min=%.3f max=%.3f zeros=%d",
        float(total_counts.min()) if len(total_counts) > 0 else np.nan,
        float(total_counts.max()) if len(total_counts) > 0 else np.nan,
        int(np.sum(total_counts == 0)),
    )

    sc.pp.normalize_total(adata, target_sum=normalize_target_sum)
    sc.pp.log1p(adata)

    if sparse.issparse(adata.X):
        adata.X = adata.X.tocsr()

    gene_means = np.asarray(adata.X.mean(axis=0)).ravel()
    safe_genes_mask = np.isfinite(gene_means)
    if not np.all(safe_genes_mask):
        logger.warning("Removing %d genes with NaN/Inf mean", int((~safe_genes_mask).sum()))
        adata = adata[:, safe_genes_mask].copy()

    if adata.n_vars == 0:
        raise RuntimeError("No genes left after base preprocessing.")

    adata.write_h5ad(base_h5ad, compression="gzip")
    write_metadata_json(meta_json, {
        "dataset_name": dataset_name,
        "source_file": input_file,
        "min_genes": min_genes,
        "min_cells": min_cells,
        "normalize_target_sum": normalize_target_sum,
        "n_obs": int(adata.n_obs),
        "n_vars": int(adata.n_vars),
    })
    logger.info("Wrote base cache: %s", base_h5ad)
    return base_h5ad


def build_or_load_hvg_pca_cache(
    base_h5ad: str,
    dataset_name: str,
    cache_dir: str,
    n_hvg: int,
    max_pcs: int,
    hvg_flavor: str = "seurat",
    random_state: int = 0,
) -> str:
    safe_mkdir(cache_dir)
    out_h5ad = os.path.join(cache_dir, f"{dataset_name}_HVG{n_hvg}_PCA{max_pcs}.h5ad")
    meta_json = os.path.join(cache_dir, f"{dataset_name}_HVG{n_hvg}_PCA{max_pcs}.meta.json")

    if os.path.exists(out_h5ad):
        logger.info("Using existing HVG/PCA cache: %s", out_h5ad)
        return out_h5ad

    logger.info("Building HVG/PCA cache for %s | n_hvg=%d | max_pcs=%d", dataset_name, n_hvg, max_pcs)
    adata = sc.read_h5ad(base_h5ad)

    n_hvg_use = min(n_hvg, adata.n_vars)
    sc.pp.highly_variable_genes(adata, n_top_genes=n_hvg_use, flavor=hvg_flavor)
    if "highly_variable" not in adata.var.columns:
        raise RuntimeError("HVG selection failed.")

    adata = adata[:, adata.var["highly_variable"]].copy()
    logger.info("After HVG selection: %s", adata.shape)

    # Match the old preprocessing pipeline
    sc.pp.scale(adata, max_value=10)
    logger.info("After scaling: %s", adata.shape)

    n_pcs_use = min(max_pcs, max(2, adata.n_vars - 1))
    sc.tl.pca(
        adata,
        svd_solver="arpack",
        n_comps=n_pcs_use,
        random_state=random_state,
    )

    logger.info("Computed PCA: X_pca shape=%s", adata.obsm["X_pca"].shape)

    adata.write_h5ad(out_h5ad, compression="gzip")
    write_metadata_json(meta_json, {
        "dataset_name": dataset_name,
        "base_h5ad": base_h5ad,
        "n_hvg": n_hvg_use,
        "max_pcs": int(n_pcs_use),
        "n_obs": int(adata.n_obs),
        "n_vars": int(adata.n_vars),
    })
    logger.info("Wrote HVG/PCA cache: %s", out_h5ad)
    return out_h5ad


def load_camp4_raw_preprocessed(
    input_file: str,
    loom_var_name_key: str,
    n_hvg: int,
    n_pcs: int,
    min_genes: int,
    min_cells: int,
) -> AnnData:
    """
    CAMP4 non-special-setting preprocessing from raw input.

    This follows the inline preprocessing logic:
    - read raw
    - filter cells / genes
    - clip negatives
    - normalize_total
    - log1p
    - ensure sparse
    - drop NaN/Inf genes
    - select HVGs
    - do NOT run PCA here
      (CAMP4 logic later will check X_pca and compute PCA with zero_center=False if needed)
    """
    logger.info("CAMP4 raw preprocessing from input: %s", input_file)

    adata = read_dataset(input_file, loom_var_name_key=loom_var_name_key)
    logger.info("CAMP4 raw read shape: %s", adata.shape)

    # filtering
    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_genes(adata, min_cells=min_cells)
    logger.info("CAMP4 after filtering: %s", adata.shape)

    # clip negatives
    if sparse.issparse(adata.X):
        if adata.X.nnz > 0:
            neg_mask = adata.X.data < 0
            n_neg = int(np.sum(neg_mask))
            if n_neg > 0:
                logger.warning("CAMP4 clipping %d negative sparse entries", n_neg)
                adata.X.data[neg_mask] = 0.0
    else:
        n_neg = int(np.sum(adata.X < 0))
        if n_neg > 0:
            logger.warning("CAMP4 clipping %d negative dense entries", n_neg)
            adata.X = np.maximum(adata.X, 0.0)

    # normalize + log1p
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    logger.info("CAMP4 normalized and log1p done")

    # ensure sparse
    if not sparse.issparse(adata.X):
        adata.X = csr_matrix(adata.X)
        logger.info("CAMP4 converted X to CSR sparse")

    # drop NaN/Inf genes
    gene_means = np.asarray(adata.X.mean(axis=0)).ravel()
    mask = np.isfinite(gene_means)
    if mask.sum() < adata.shape[1]:
        logger.warning("CAMP4 dropping %d bad genes", int(adata.shape[1] - mask.sum()))
        adata = adata[:, mask].copy()

    if adata.n_vars == 0:
        raise RuntimeError("CAMP4 preprocessing left zero genes.")

    # HVG
    n_hvg_use = min(n_hvg, adata.n_vars)
    sc.pp.highly_variable_genes(adata, n_top_genes=n_hvg_use, flavor="seurat")
    if "highly_variable" not in adata.var.columns:
        raise RuntimeError("CAMP4 HVG selection failed.")

    hvg_count = int(adata.var["highly_variable"].sum())
    logger.info("CAMP4 selected %d HVGs", hvg_count)
    adata = adata[:, adata.var["highly_variable"]].copy()
    logger.info("CAMP4 after HVG: %s", adata.shape)

    # build PCA/diffmap using the requested n_pcs
    n_comps_use = min(n_pcs, max(2, adata.n_vars - 1))

    if "X_pca" in adata.obsm and adata.obsm["X_pca"].shape[1] >= n_comps_use:
        adata.obsm["X_pca"] = np.asarray(adata.obsm["X_pca"][:, :n_comps_use], dtype=np.float32)
        logger.info("CAMP4 reused existing X_pca and sliced to shape=%s", adata.obsm["X_pca"].shape)
    else:
        sc.tl.pca(
            adata,
            n_comps=n_comps_use,
            svd_solver="arpack",
            zero_center=False,
        )
        adata.obsm["X_pca"] = np.asarray(adata.obsm["X_pca"][:, :n_comps_use], dtype=np.float32)
        logger.info("CAMP4 computed X_pca with requested n_pcs=%d | shape=%s", n_comps_use, adata.obsm["X_pca"].shape)

    sc.pp.neighbors(adata, use_rep="X_pca")
    sc.tl.diffmap(adata)
    logger.info("CAMP4 computed neighbors and X_diffmap for metrics")

    # build global PCA/diffmap for downstream metrics
    #if "X_pca" not in adata.obsm:
    #    n_comps_use = min(50, max(2, adata.n_vars - 1))
    #    sc.tl.pca(
    #        adata,
    #        n_comps=n_comps_use,
    #        svd_solver="arpack",
    #        zero_center=False,
    #    )
    #    logger.info("CAMP4 computed X_pca for metrics | shape=%s", adata.obsm["X_pca"].shape)

    #sc.pp.neighbors(adata, use_rep="X_pca")
    #sc.tl.diffmap(adata)
    #logger.info("CAMP4 computed neighbors and X_diffmap for metrics")


    # important: no PCA here
    return adata

def load_cache_with_sliced_pca(hvg_pca_h5ad: str, n_pcs: int) -> AnnData:
    adata = sc.read_h5ad(hvg_pca_h5ad)
    if "X_pca" not in adata.obsm:
        raise RuntimeError(f"X_pca missing in cached file: {hvg_pca_h5ad}")

    max_cached = adata.obsm["X_pca"].shape[1]
    n_pcs_use = min(n_pcs, max_cached)
    adata.obsm["X_pca"] = np.asarray(adata.obsm["X_pca"][:, :n_pcs_use], dtype=np.float32)

    sc.pp.neighbors(adata, use_rep="X_pca")
    sc.tl.diffmap(adata)

    # CAMP2 RECOMB-style cosine similarity matrix
    X_unit = normalize(adata.obsm["X_pca"], axis=1, copy=True)
    adata.obsm["X_unit"] = X_unit
    adata.obsm["X_simi_cos"] = X_unit.dot(X_unit.T)

    # CAMP3 RECOMB-style adaptive Gaussian kernel
    adata.obsm["X_simi_adg"] = compute_adaptive_kernel(
        adata.obsm["X_pca"],
        n_neighbors=15,
        symmetrize=True,
    )

    logger.info(
        "Loaded cache %s | sliced PCA to n_pcs=%d | recomputed neighbors/diffmap | built X_simi_cos and X_simi_adg",
        hvg_pca_h5ad, n_pcs_use
    )
    return adata


# =========================================================
# CAMP shared core
# =========================================================
def compute_lightweight_coreset_q(
    adata: AnnData,
    rep_key: Optional[str] = None,
    layer: Optional[str] = None
) -> np.ndarray:
    if rep_key is not None:
        X = adata.obsm[rep_key]
    elif layer is not None:
        X = adata.layers[layer]
    else:
        X = adata.X

    if sparse.issparse(X):
        X_csr = X.tocsr(copy=False)
        mu = np.asarray(X_csr.mean(axis=0)).ravel()
        x_norm_sq = np.asarray(X_csr.power(2).sum(axis=1)).ravel()
        mu_norm_sq = float(mu @ mu)
        prod = X_csr @ mu
        d_sq = x_norm_sq + mu_norm_sq - 2.0 * prod
    else:
        X = np.asarray(X, dtype=np.float64)
        mu = X.mean(axis=0)
        d_sq = np.sum((X - mu) ** 2, axis=1)

    n = adata.n_obs
    denom = d_sq.sum()
    if denom == 0.0:
        logger.warning("All points coincide; returning uniform q.")
        return np.full(n, 1.0 / n)

    q = 0.5 * (1.0 / n) + 0.5 * (d_sq / denom)
    q /= q.sum()
    return q


def create_m_list(adata: AnnData, gamma_list: List[int], rep_key: str = "X_pca") -> Tuple[List[int], np.ndarray]:
    start = time.time()
    sample_scores = compute_lightweight_coreset_q(adata, rep_key=rep_key)
    adata.obs["q"] = sample_scores
    logger.info("Computed q(x) in %.2fs; q.sum()=%.6f", time.time() - start, float(sample_scores.sum()))
    n_cells = adata.n_obs
    m_list = [int(n_cells / g) for g in gamma_list]
    return m_list, sample_scores


def create_coreset_unselected(adata: AnnData, m: int, sample_scores: np.ndarray) -> Tuple[AnnData, AnnData]:
    all_idx = np.arange(adata.n_obs)
    idx = np.random.choice(adata.n_obs, size=m, replace=False, p=sample_scores)
    coreset = adata[idx].copy()
    unselected = adata[np.setdiff1d(all_idx, idx)].copy()
    return coreset, unselected


# =========================================================
# CAMP4 helpers
# =========================================================
def assign_from_fixed_archetypes(
    subset: AnnData,
    seed_names: List[str],
    reduction_key: str = "X_pca",
    n_neighbors: int = 30,
    graph_construction: str = "union",
    chunk_size: int = 20_000,
) -> np.ndarray:
    g = build_graph.SEACellGraph(subset, reduction_key, verbose=False)

    try:
        K = g.rbf(n_neighbors=n_neighbors, graph_construction=graph_construction)
    except TypeError:
        g.n_neighbors = n_neighbors
        K = g.rbf(graph_construction=graph_construction)

    K = K.tocsr()
    n_sub = subset.n_obs

    m_seeds = len(seed_names)
    seed_idx = [subset.obs_names.get_loc(s) for s in seed_names]

    data = np.ones(m_seeds, dtype=np.float32)
    rows = np.array(seed_idx, dtype=int)
    cols = np.arange(m_seeds, dtype=int)
    B = csr_matrix((data, (rows, cols)), shape=(n_sub, m_seeds))

    labels = np.empty(n_sub, dtype=int)
    for start in range(0, n_sub, chunk_size):
        end = min(start + chunk_size, n_sub)
        K_block = K[start:end, :]
        A_block = K_block.dot(B).toarray()
        labels[start:end] = A_block.argmax(axis=1)

    return labels


# =========================================================
# Literal RECOMB CAMP4 runner
# =========================================================
def create_m_list_recomb_camp4(
    adata: AnnData,
    gamma_list: List[int],
) -> Tuple[List[int], np.ndarray]:
    start = time.time()
    q = compute_lightweight_coreset_q(adata, rep_key="X_pca")
    adata.obs["q"] = q
    logger.info("computed q(x) in %.1fs; q.sum()=%.4f", time.time() - start, q.sum())
    n_cells = adata.n_obs
    m_list = [max(1, int(n_cells / g)) for g in gamma_list]
    return m_list, q


def sample_coreset_recomb_camp4(
    subset: AnnData,
    m: int,
    probs: np.ndarray,
) -> AnnData:
    idx = np.random.choice(subset.n_obs, size=m, replace=False, p=probs)
    return subset[idx].copy()


def run_camp4_recomb_exact(
    adata: AnnData,
    gamma_list: List[int],
    output_csv: str,
    reduction_key: str = "X_pca",
    annotations: Optional[str] = None,
    min_metacells: int = 1,
):
    """
    Literal RECOMB CAMP4 logic.
    """
    adata = adata.copy()

    if annotations is None or annotations not in adata.obs:
        annotations = "SEACell_batch"
        adata.obs[annotations] = "allcells"

    m_list, _ = create_m_list_recomb_camp4(adata, gamma_list)

    cell_membership = pd.DataFrame(index=adata.obs_names)
    seed_flag = pd.DataFrame(index=adata.obs_names)

    for gamma, m in zip(gamma_list, m_list):
        col = str(gamma)
        seed_col = f"{col}_is_seed"
        cell_membership[col] = None
        seed_flag[seed_col] = False

        logger.info("=== gamma=%s, m≈%s ===", gamma, m)
        t_gamma = time.time()

        for anno in adata.obs[annotations].unique():
            subset = adata[adata.obs[annotations] == anno].copy()

            if reduction_key not in subset.obsm:
                sc.tl.pca(subset, n_comps=50, zero_center=False)

            if subset.n_obs <= min_metacells:
                labels = [f"{anno}-SEACell-1"] * subset.n_obs
                cell_membership.loc[subset.obs_names, col] = labels
                continue

            if "q" in subset.obs:
                q_sub = subset.obs["q"].to_numpy()
            else:
                q_sub = compute_lightweight_coreset_q(subset, rep_key=reduction_key)
                subset.obs["q"] = q_sub

            m_here = min(m, subset.n_obs)
            coreset = sample_coreset_recomb_camp4(subset, m_here, q_sub)
            seed_names = list(coreset.obs_names)

            labels_idx = assign_from_fixed_archetypes(
                subset,
                seed_names=seed_names,
                reduction_key=reduction_key,
                n_neighbors=30,
                graph_construction="union",
                chunk_size=20000,
            )

            final_labels = [f"seed{lab}-{gamma}-{anno}" for lab in labels_idx]

            cell_membership.loc[subset.obs_names, col] = final_labels
            seed_flag.loc[seed_names, seed_col] = True

            logger.info("%s: sampled %d seeds (final archetypes)", anno, len(seed_names))
            logger.info("%s: done (%d cells)", anno, subset.n_obs)

        logger.info("gamma=%s done in %.1fs", gamma, time.time() - t_gamma)

    out = pd.concat([cell_membership, seed_flag], axis=1)
    out.to_csv(output_csv)
    logger.info("Saved partitions: %s", output_csv)


# =========================================================
# Variant runner
# =========================================================
def run_camp_variant(
    adata: AnnData,
    variant: str,
    gamma_list: List[int],
    output_csv: str,
    annotations: Optional[str] = None,
    min_metacells: int = 1,
    random_state: int = 0,
):
    np.random.seed(random_state)
    adata = adata.copy()

    if annotations is None or annotations not in adata.obs:
        annotations = "SEACell_batch"
        adata.obs[annotations] = "allcells"

    if variant == "camp4":
        sample_scores = compute_lightweight_coreset_q(adata, rep_key="X_pca")
        adata.obs["q"] = sample_scores
        m_list = [max(1, int(adata.n_obs / g)) for g in gamma_list]
    elif variant == "camp3":
        logger.info("camp3: starting q(x) from X_simi_adg")
        sample_scores = compute_lightweight_coreset_q(adata, rep_key="X_simi_adg")
        logger.info("camp3: finished q(x) from X_simi_adg")
        adata.obs["q"] = sample_scores
        m_list = [int(adata.n_obs / g) for g in gamma_list]
    elif variant == "camp2":
        sample_scores = compute_lightweight_coreset_q(adata, rep_key="X_simi_cos")
        adata.obs["q"] = sample_scores
        m_list = [int(adata.n_obs / g) for g in gamma_list]
    else:  # camp1
        sample_scores = compute_lightweight_coreset_q(adata, rep_key="X_pca")
        adata.obs["q"] = sample_scores
        m_list = [int(adata.n_obs / g) for g in gamma_list]

    logger.info("Computed q(x); q.sum()=%.6f", float(sample_scores.sum()))

    cell_membership = pd.DataFrame(index=adata.obs_names)
    seed_flag = pd.DataFrame(index=adata.obs_names)

    for gamma, m in zip(gamma_list, m_list):
        logger.info("Running %s | gamma=%s | m=%s", variant, gamma, m)
        col = str(gamma)
        seed_col = f"{col}_is_seed"

        cell_membership[col] = None
        seed_flag[seed_col] = False

        for anno in adata.obs[annotations].unique():
            subset = adata[adata.obs[annotations] == anno].copy()

            if subset.n_obs <= min_metacells:
                labels = [f"{anno}-SEACell-1"] * subset.n_obs
                seeds = []
            else:
                if variant == "camp3":
                    subset_rep_key = "X_simi_adg"
                elif variant == "camp2":
                    subset_rep_key = "X_simi_cos"
                else:
                    subset_rep_key = "X_pca"

                scores = (
                    subset.obs["q"].values
                    if "q" in subset.obs
                    else compute_lightweight_coreset_q(subset, rep_key=subset_rep_key)
                )
                subset.obs["q"] = scores

                if variant == "camp4":
                    m_here = min(m, subset.n_obs)
                else:
                    m_here = m

                coreset, _ = create_coreset_unselected(subset, m_here, scores)
                seeds = list(coreset.obs_names)

                if variant == "camp1":
                    X_emb = subset.obsm["X_pca"]
                    seed_idx_local = [subset.obs_names.get_loc(x) for x in coreset.obs_names]
                    seed_emb = X_emb[seed_idx_local]
                    km = KMeans(
                        n_clusters=m_here,
                        init=seed_emb,
                        n_init=1,
                        max_iter=10,
                        algorithm="lloyd",
                    ).fit(X_emb)
                    labels_idx = km.labels_

                elif variant == "camp2":
                    subset_idx = np.fromiter(
                        (adata.obs_names.get_loc(c) for c in subset.obs_names),
                        dtype=np.int64
                    )
                    seed_idx = np.fromiter(
                        (adata.obs_names.get_loc(c) for c in coreset.obs_names),
                        dtype=np.int64
                    )

                    sim_sub = adata.obsm["X_simi_cos"][np.ix_(subset_idx, seed_idx)]
                    labels_idx = np.asarray(np.argmax(sim_sub, axis=1)).ravel()

                elif variant == "camp3":
                    subset_idx = np.fromiter(
                        (adata.obs_names.get_loc(c) for c in subset.obs_names),
                        dtype=np.int64
                    )
                    seed_idx = np.fromiter(
                        (adata.obs_names.get_loc(c) for c in coreset.obs_names),
                        dtype=np.int64
                    )

                    sim_sub = adata.obsm["X_simi_adg"][subset_idx][:, seed_idx]

                    if sparse.issparse(sim_sub):
                        sim_sub = sim_sub.toarray()

                    labels_idx = np.asarray(np.argmax(sim_sub, axis=1)).ravel()

                elif variant == "camp4":
                    labels_idx = assign_from_fixed_archetypes(
                        subset,
                        seed_names=seeds,
                        reduction_key="X_pca",
                        n_neighbors=30,
                        graph_construction="union",
                        chunk_size=20_000,
                    )

                else:
                    raise ValueError(f"Unknown variant: {variant}")

                labels = [f"seed{lab}-{gamma}-{anno}" for lab in labels_idx]

            cell_membership.loc[subset.obs_names, col] = labels
            if len(seeds) > 0:
                seed_flag.loc[seeds, seed_col] = True

    out = pd.concat([cell_membership, seed_flag], axis=1)
    out.to_csv(output_csv)
    logger.info("Saved partitions: %s", output_csv)


# =========================================================
# Metrics
# =========================================================
def compute_purity_from_labels(cell_labels: pd.Series) -> float:
    counts = cell_labels.value_counts(dropna=False)
    if len(counts) == 0:
        return np.nan
    return float(counts.max()) / float(counts.sum())


def compute_label_entropy(cell_labels: pd.Series) -> float:
    counts = cell_labels.value_counts(dropna=False).values.astype(float)
    if counts.sum() == 0:
        return np.nan
    p = counts / counts.sum()
    return float(scipy_entropy(p, base=2))


def compute_compactness(diff_coords: np.ndarray) -> float:
    # higher is better because this is negative variance
    return float(-np.var(diff_coords, axis=0).mean())


def compute_INV(expr_block) -> float:
    X = expr_block.toarray() if sparse.issparse(expr_block) else np.asarray(expr_block)
    gene_var = np.var(X, axis=0)
    gene_mean = np.mean(X, axis=0)
    inv = np.percentile(gene_var / (gene_mean + 1e-8), 95)
    return float(inv)


def compute_sc_ratio(compactness: float, separation: float) -> float:
    denom = np.sqrt(max(1e-12, -compactness))
    return float(separation / denom)


def ecdf_xy(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.array([]), np.array([])
    x = np.sort(vals)
    y = np.arange(1, x.size + 1) / x.size
    return x, y


def compute_metrics_for_partition_file(
    adata: AnnData,
    partition_csv: str,
    dataset_name: str,
    variant: str,
    variation_mode: str,
    variation_value: int,
    celltype_key: Optional[str],
    out_raw_csv: str,
    out_summary_csv: str,
):
    part = pd.read_csv(partition_csv, index_col=0)
    gamma_cols = [c for c in part.columns if not c.endswith("_is_seed")]
    if len(gamma_cols) == 0:
        raise RuntimeError(f"No gamma columns found in {partition_csv}")

    # --- RECOMB-style alignment ---
    common_cells = adata.obs_names.intersection(part.index)
    ad = adata[common_cells].copy()
    part = part.loc[common_cells].copy()


    if "X_diffmap" not in adata.obsm:
        if "X_pca" not in adata.obsm:
            raise RuntimeError("X_diffmap missing and X_pca also missing; cannot compute metrics.")
        sc.pp.neighbors(adata, use_rep="X_pca")
        sc.tl.diffmap(adata)


    X_diff = ad.obsm["X_diffmap"]
    X_diff_dense = X_diff if not sparse.issparse(X_diff) else X_diff.toarray()
    X_expr = ad.X

    raw_rows = []
    summary_rows = []

    for gamma_col in gamma_cols:
        ad.obs["metacell"] = part[gamma_col].astype(str)
        grouped = ad.obs.groupby("metacell", observed=True)

        centroids = {}
        for metacell_id, group in grouped:
            ix = ad.obs.index.get_indexer(group.index)
            diff_coords = X_diff_dense[ix]
            centroids[metacell_id] = diff_coords.mean(axis=0)

        mc_ids = list(centroids.keys())
        C = np.vstack([centroids[mid] for mid in mc_ids])
        D = pairwise_distances(C, C)
        np.fill_diagonal(D, np.inf)
        nearest_sep = np.min(D, axis=1)
        sep_map = {mid: float(nearest_sep[j]) for j, mid in enumerate(mc_ids)}

        per_mc = []
        for metacell_id, group in grouped:
            ix = ad.obs.index.get_indexer(group.index)
            diff_coords = X_diff_dense[ix]
            expr_block = X_expr[ix, :]

            compactness = compute_compactness(diff_coords)
            separation = sep_map[metacell_id]
            sc_ratio = compute_sc_ratio(compactness, separation)

            row = {
                "dataset": dataset_name,
                "variant": variant,
                "variation_mode": variation_mode,
                "variation_value": variation_value,
                "gamma": int(gamma_col),
                "metacell_id": metacell_id,
                "compactness": compactness,
                "separation": separation,
                "sc_ratio": sc_ratio,
                "INV": compute_INV(expr_block),
            }

            if celltype_key is not None and celltype_key in ad.obs.columns:
                label_series = ad.obs.loc[group.index, celltype_key]
                row["purity"] = compute_purity_from_labels(label_series)
                row["label_entropy"] = compute_label_entropy(label_series)
            else:
                row["purity"] = np.nan
                row["label_entropy"] = np.nan

            per_mc.append(row)

        raw_df = pd.DataFrame(per_mc)
        raw_rows.append(raw_df)

        summary_row = {
            "dataset": dataset_name,
            "variant": variant,
            "variation_mode": variation_mode,
            "variation_value": variation_value,
            "gamma": int(gamma_col),

            "purity_mean": raw_df["purity"].mean(skipna=True),
            "purity_median": raw_df["purity"].median(skipna=True),
            "purity_std": raw_df["purity"].std(skipna=True),
            "purity_q25": raw_df["purity"].quantile(0.25),
            "purity_q75": raw_df["purity"].quantile(0.75),

            "entropy_mean": raw_df["label_entropy"].mean(skipna=True),
            "entropy_median": raw_df["label_entropy"].median(skipna=True),
            "entropy_std": raw_df["label_entropy"].std(skipna=True),
            "entropy_q25": raw_df["label_entropy"].quantile(0.25),
            "entropy_q75": raw_df["label_entropy"].quantile(0.75),

            "compactness_mean": raw_df["compactness"].mean(skipna=True),
            "compactness_median": raw_df["compactness"].median(skipna=True),
            "compactness_std": raw_df["compactness"].std(skipna=True),
            "compactness_q25": raw_df["compactness"].quantile(0.25),
            "compactness_q75": raw_df["compactness"].quantile(0.75),

            "separation_mean": raw_df["separation"].mean(skipna=True),
            "separation_median": raw_df["separation"].median(skipna=True),
            "separation_std": raw_df["separation"].std(skipna=True),
            "separation_q25": raw_df["separation"].quantile(0.25),
            "separation_q75": raw_df["separation"].quantile(0.75),

            "sc_ratio_mean": raw_df["sc_ratio"].mean(skipna=True),
            "sc_ratio_median": raw_df["sc_ratio"].median(skipna=True),
            "sc_ratio_std": raw_df["sc_ratio"].std(skipna=True),
            "sc_ratio_q25": raw_df["sc_ratio"].quantile(0.25),
            "sc_ratio_q75": raw_df["sc_ratio"].quantile(0.75),

            "INV_mean": raw_df["INV"].mean(skipna=True),
            "INV_median": raw_df["INV"].median(skipna=True),
            "INV_std": raw_df["INV"].std(skipna=True),
            "INV_q25": raw_df["INV"].quantile(0.25),
            "INV_q75": raw_df["INV"].quantile(0.75),
        }
        summary_rows.append(summary_row)

    raw_all = pd.concat(raw_rows, ignore_index=True)
    summary_all = pd.DataFrame(summary_rows)

    raw_all.to_csv(out_raw_csv, index=False)
    summary_all.to_csv(out_summary_csv, index=False)

    logger.info("Saved raw metrics: %s", out_raw_csv)
    logger.info("Saved summary metrics: %s", out_summary_csv)


# =========================================================
# Plotting
# =========================================================
def save_shared_legend_png(out_png: str, variants: List[str], dpi: int = 600):
    handles = []
    for v in variants:
        handles.append(
            Line2D(
                [0], [0],
                color=variant_to_color(v),
                marker="o",
                linewidth=2.2,
                markersize=6,
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
        fontsize=12,
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

    fig, ax = plt.subplots(figsize=(6.2, 4.8))

    for variant in [v for v in VARIANT_ORDER if v in set(sub["variant"])]:
        ss = sub[sub["variant"] == variant].sort_values("gamma")

        x = ss["gamma"].values.astype(float)
        y = ss[metric_col].values.astype(float)
        color = variant_to_color(variant)

        ax.plot(
            x,
            y,
            marker="o",
            linewidth=2.2,
            markersize=5.5,
            color=color,
        )

    if variation_mode == "hvg":
        title_suffix = f"HVG={variation_value}"
    else:
        title_suffix = f"PCs={variation_value}"

    ax.set_title(f"{dataset_name} | {title_suffix}", fontsize=14)
    ax.set_xlabel("gamma", fontsize=12)
    ax.set_ylabel(PRETTY_METRIC_LABELS.get(metric_col, metric_col), fontsize=12)
    ax.grid(False)
    ax.tick_params(labelsize=11)

    fig.tight_layout()
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved plot: %s", out_png)


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

    for variant in [v for v in VARIANT_ORDER if v in set(sub["variant"])]:
        vals = sub.loc[sub["variant"] == variant, "sc_ratio"].values
        x, y = ecdf_xy(vals)
        if x.size == 0:
            continue
        ax.step(
            x,
            y,
            where="post",
            linewidth=2.2,
            color=variant_to_color(variant),
        )

    x_label_name = "n_hvg" if variation_mode == "hvg" else "n_pcs"
    ax.set_title(
        f"{dataset_name} | {x_label_name}={variation_value} | gamma={gamma}",
        fontsize=14
    )
    ax.set_xlabel("SC ratio", fontsize=12)
    ax.set_ylabel("ECDF", fontsize=12)
    ax.grid(False)
    ax.tick_params(labelsize=11)

    fig.tight_layout()
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved ECDF plot: %s", out_png)


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

    if ecdf_gammas is None:
        ecdf_gammas = [50, 200, 500]

    ecdf_gammas = [g for g in ecdf_gammas if g in gammas]

    line_dir = os.path.join(out_dir, "line_plots")
    safe_mkdir(line_dir)

    for metric in metrics_to_plot:
        metric_dir = os.path.join(line_dir, metric)
        safe_mkdir(metric_dir)
        for vv in variation_values:
            out_png = os.path.join(
                metric_dir,
                f"{dataset_name}_{variation_mode}_{metric}_value{vv}.png"
            )
            plot_metric_lines_single_variation(
                summary_df=summary_df,
                dataset_name=dataset_name,
                variation_mode=variation_mode,
                metric_col=metric,
                variation_value=vv,
                out_png=out_png,
                dpi=dpi,
            )

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
# Experiment driver
# =========================================================
def one_dataset_one_variation(
    input_file: str,
    dataset_name: str,
    celltype_key: Optional[str],
    output_root: str,
    variation_mode: str,
    variation_values: List[int],
    fixed_hvg: int,
    fixed_pcs: int,
    variants: List[str],
    gamma_list: List[int],
    loom_var_name_key: str,
    min_genes: int,
    min_cells: int,
    random_state: int,
    dpi: int,
    ecdf_gammas: Optional[List[int]] = None,
):
    logger.info("=" * 80)
    logger.info("Dataset=%s | mode=%s", dataset_name, variation_mode)
    logger.info("=" * 80)

    dataset_dir = os.path.join(output_root, dataset_name, variation_mode)
    cache_dir = os.path.join(output_root, dataset_name, "cache")
    prep_dir = os.path.join(dataset_dir, "preprocessed")
    part_dir = os.path.join(dataset_dir, "partitions")
    metrics_dir = os.path.join(dataset_dir, "metrics")
    plots_dir = os.path.join(dataset_dir, "plots")

    for d in [dataset_dir, cache_dir, prep_dir, part_dir, metrics_dir, plots_dir]:
        safe_mkdir(d)

    # Hardcoded special mode:
    # CAMP4 uses blish_covid_processed.h5ad exactly when HVG=2000 and PCA=50.
    use_direct_recomb_mode = (
        variation_mode == "hvg"
        and sorted(set(int(v) for v in variation_values)) == [2000]
        and int(fixed_pcs) == 50
        and os.path.exists(CAMP4_DIRECT_PROCESSED_FILE)
    )


    if use_direct_recomb_mode:
        logger.info(
            "Using mixed mode for %s | camp1/2/3 use cached pipeline from %s | camp4 uses hardcoded processed file for the special setting and raw self-preprocessing otherwise",
            dataset_name,
            input_file,
        )
    else:
        logger.info(
            "Using mixed mode for %s | camp1/2/3 use cached pipeline from %s | camp4 uses raw self-preprocessing",
            dataset_name,
            input_file,
        )

    #if use_direct_recomb_mode:
    #    logger.info(
    #        "Using mixed mode for %s | camp4 uses hardcoded processed input=%s | camp1/2/3 use cached pipeline from %s",
    #        dataset_name,
    #        CAMP4_DIRECT_PROCESSED_FILE,
    #        input_file,
    #    )
    #else:
    #    logger.info(
    #        "Using cached pipeline for all variants from %s",
    #        input_file,
    #    )

    # Always prepare cache for camp1/2/3 from the raw/seu input
    base_h5ad = build_or_load_base_cache(
        input_file=input_file,
        dataset_name=dataset_name,
        cache_dir=cache_dir,
        loom_var_name_key=loom_var_name_key,
        min_genes=min_genes,
        min_cells=min_cells,
    )

    if variation_mode == "hvg":
        max_pcs_needed = fixed_pcs
        hvg_values_needed = sorted(set(int(v) for v in variation_values))
    elif variation_mode == "pca":
        max_pcs_needed = max(int(v) for v in variation_values)
        hvg_values_needed = [int(fixed_hvg)]
    else:
        raise ValueError("variation_mode must be 'hvg' or 'pca'")

    hvg_cache_map = {}
    for n_hvg in hvg_values_needed:
        hvg_cache_map[n_hvg] = build_or_load_hvg_pca_cache(
            base_h5ad=base_h5ad,
            dataset_name=dataset_name,
            cache_dir=cache_dir,
            n_hvg=n_hvg,
            max_pcs=max_pcs_needed,
            hvg_flavor="seurat",
            random_state=random_state,
        )

    all_summary_paths = []
    all_raw_paths = []

    for value in variation_values:
        if variation_mode == "hvg":
            n_hvg = int(value)
            n_pcs = int(fixed_pcs)
        else:
            n_hvg = int(fixed_hvg)
            n_pcs = int(value)

        cache_h5ad = hvg_cache_map[n_hvg]

        for variant in variants:
            logger.info("Running %s on %s | %s=%d", variant, dataset_name, variation_mode, value)

            # Only CAMP4 gets the hardcoded processed file, and only at HVG=2000, PCA=50
            use_direct_for_this_variant = (
                variant == "camp4"
                #and variation_mode == "hvg"
                and n_hvg == 2000
                and n_pcs == 50
                and os.path.exists(CAMP4_DIRECT_PROCESSED_FILE)
            )

            if use_direct_for_this_variant:
                adata = load_recomb_processed_direct(
                    input_file=CAMP4_DIRECT_PROCESSED_FILE,
                    loom_var_name_key=loom_var_name_key,
                    n_pcs=n_pcs,
                )
                source_mode = "direct_recomb_processed"
            elif variant == "camp4":
                adata = load_camp4_raw_preprocessed(
                    input_file=input_file,
                    loom_var_name_key=loom_var_name_key,
                    n_hvg=n_hvg,
                    n_pcs=n_pcs,
                    min_genes=min_genes,
                    min_cells=min_cells,
                )
                source_mode = "camp4_raw_preprocessed"
            else:
                adata = load_cache_with_sliced_pca(cache_h5ad, n_pcs=n_pcs)
                source_mode = "cached_pipeline"

            #if use_direct_for_this_variant:
            #    adata = load_recomb_processed_direct(
            #        input_file=CAMP4_DIRECT_PROCESSED_FILE,
            #        loom_var_name_key=loom_var_name_key,
            #        n_pcs=n_pcs,
            #        #random_state=random_state,
            #    )
            #    source_mode = "direct_recomb_processed"
            #else:
            #    adata = load_cache_with_sliced_pca(cache_h5ad, n_pcs=n_pcs)
            #    source_mode = "cached_pipeline"

            prep_h5ad = os.path.join(
                prep_dir,
                f"{dataset_name}_{variant}_{variation_mode}_{value}_prep.h5ad"
            )
            if not os.path.exists(prep_h5ad):
                adata.write_h5ad(prep_h5ad, compression="gzip")

            prep_meta = os.path.join(
                prep_dir,
                f"{dataset_name}_{variant}_{variation_mode}_{value}_prep.meta.json"
            )
            write_metadata_json(prep_meta, {
                "dataset": dataset_name,
                "variant": variant,
                "variation_mode": variation_mode,
                "variation_value": int(value),
                "n_hvg": int(n_hvg),
                "n_pcs": int(n_pcs),
                "celltype_key": celltype_key,
                "gamma_list": [int(g) for g in gamma_list],
                "variants": variants,
                "source_mode": source_mode,
                "raw_input_file": input_file,
                "camp4_direct_processed_file": CAMP4_DIRECT_PROCESSED_FILE if use_direct_for_this_variant else None,
            })

            part_csv = os.path.join(part_dir, f"{dataset_name}_{variant}_{variation_mode}_{value}_partitions.csv")
            run_meta = os.path.join(part_dir, f"{dataset_name}_{variant}_{variation_mode}_{value}_run.meta.json")

            if not os.path.exists(part_csv):
                if use_direct_for_this_variant:
                    logger.info("Running literal RECOMB CAMP4 exact mode from hardcoded processed file")
                    np.random.seed(random_state)
                    run_camp4_recomb_exact(
                        adata=adata,
                        gamma_list=gamma_list,
                        output_csv=part_csv,
                        reduction_key="X_pca",
                        annotations=None,
                        min_metacells=1,
                    )
                else:
                    run_camp_variant(
                        adata=adata,
                        variant=variant,
                        gamma_list=gamma_list,
                        output_csv=part_csv,
                        annotations=None,
                        min_metacells=1,
                        random_state=random_state,
                    )

            write_metadata_json(run_meta, {
                "dataset": dataset_name,
                "variant": variant,
                "variation_mode": variation_mode,
                "variation_value": int(value),
                "n_hvg": int(n_hvg),
                "n_pcs": int(n_pcs),
                "partition_csv": part_csv,
                "prep_h5ad": prep_h5ad,
                "source_mode": source_mode,
                "celltype_key": celltype_key,
                "gamma_list": [int(g) for g in gamma_list],
                "raw_input_file": input_file,
                "camp4_direct_processed_file": CAMP4_DIRECT_PROCESSED_FILE if use_direct_for_this_variant else None,
            })

            raw_metrics_csv = os.path.join(metrics_dir, f"{dataset_name}_{variant}_{variation_mode}_{value}_metrics_raw.csv")
            summary_metrics_csv = os.path.join(metrics_dir, f"{dataset_name}_{variant}_{variation_mode}_{value}_metrics_summary.csv")

            if not (os.path.exists(raw_metrics_csv) and os.path.exists(summary_metrics_csv)):
                compute_metrics_for_partition_file(
                    adata=adata,
                    partition_csv=part_csv,
                    dataset_name=dataset_name,
                    variant=variant,
                    variation_mode=variation_mode,
                    variation_value=value,
                    celltype_key=celltype_key,
                    out_raw_csv=raw_metrics_csv,
                    out_summary_csv=summary_metrics_csv,
                )

            all_raw_paths.append(raw_metrics_csv)
            all_summary_paths.append(summary_metrics_csv)

    summary_frames = [pd.read_csv(p) for p in all_summary_paths if os.path.exists(p)]
    raw_frames = [pd.read_csv(p) for p in all_raw_paths if os.path.exists(p)]

    if len(summary_frames) > 0:
        combined_summary = pd.concat(summary_frames, ignore_index=True)
        combined_summary_csv = os.path.join(metrics_dir, f"{dataset_name}_{variation_mode}_ALL_summary.csv")
        combined_summary.to_csv(combined_summary_csv, index=False)
        logger.info("Saved combined summary: %s", combined_summary_csv)
    else:
        combined_summary = pd.DataFrame()

    if len(raw_frames) > 0:
        combined_raw = pd.concat(raw_frames, ignore_index=True)
        combined_raw_csv = os.path.join(metrics_dir, f"{dataset_name}_{variation_mode}_ALL_raw.csv")
        combined_raw.to_csv(combined_raw_csv, index=False)
        logger.info("Saved combined raw metrics: %s", combined_raw_csv)
    else:
        combined_raw = pd.DataFrame()

    if not combined_summary.empty and not combined_raw.empty:
        make_all_plots_for_dataset_variation(
            summary_df=combined_summary,
            raw_df=combined_raw,
            out_dir=plots_dir,
            dataset_name=dataset_name,
            variation_mode=variation_mode,
            dpi=dpi,
            ecdf_gammas=ecdf_gammas,
        )

def previous_one_dataset_one_variation(
    input_file: str,
    dataset_name: str,
    celltype_key: Optional[str],
    output_root: str,
    variation_mode: str,
    variation_values: List[int],
    fixed_hvg: int,
    fixed_pcs: int,
    variants: List[str],
    gamma_list: List[int],
    loom_var_name_key: str,
    min_genes: int,
    min_cells: int,
    random_state: int,
    dpi: int,
    ecdf_gammas: Optional[List[int]] = None,
    camp4_direct_input: str = "",
):
    logger.info("=" * 80)
    logger.info("Dataset=%s | mode=%s", dataset_name, variation_mode)
    logger.info("=" * 80)

    dataset_dir = os.path.join(output_root, dataset_name, variation_mode)
    cache_dir = os.path.join(output_root, dataset_name, "cache")
    prep_dir = os.path.join(dataset_dir, "preprocessed")
    part_dir = os.path.join(dataset_dir, "partitions")
    metrics_dir = os.path.join(dataset_dir, "metrics")
    plots_dir = os.path.join(dataset_dir, "plots")

    for d in [dataset_dir, cache_dir, prep_dir, part_dir, metrics_dir, plots_dir]:
        safe_mkdir(d)

    # special mode:
    # camp1/2/3 -> cache from input_file
    # camp4     -> direct load from camp4_direct_input
    use_direct_camp4_mode = (
        bool(camp4_direct_input)
        and variation_mode == "hvg"
        and sorted(set(int(v) for v in variation_values)) == [2000]
        and int(fixed_pcs) == 50
    )

    if use_direct_camp4_mode:
        logger.info(
            "Using mixed mode for %s | camp1/2/3 use cached pipeline from %s | camp4 uses direct processed input %s",
            dataset_name,
            input_file,
            camp4_direct_input,
        )

    # Always prepare cache from the main input file for camp1/2/3
    base_h5ad = build_or_load_base_cache(
        input_file=input_file,
        dataset_name=dataset_name,
        cache_dir=cache_dir,
        loom_var_name_key=loom_var_name_key,
        min_genes=min_genes,
        min_cells=min_cells,
    )

    if variation_mode == "hvg":
        max_pcs_needed = fixed_pcs
        hvg_values_needed = sorted(set(int(v) for v in variation_values))
    elif variation_mode == "pca":
        max_pcs_needed = max(int(v) for v in variation_values)
        hvg_values_needed = [int(fixed_hvg)]
    else:
        raise ValueError("variation_mode must be 'hvg' or 'pca'")

    hvg_cache_map = {}
    for n_hvg in hvg_values_needed:
        hvg_cache_map[n_hvg] = build_or_load_hvg_pca_cache(
            base_h5ad=base_h5ad,
            dataset_name=dataset_name,
            cache_dir=cache_dir,
            n_hvg=n_hvg,
            max_pcs=max_pcs_needed,
            hvg_flavor="seurat",
            random_state=random_state,
        )

    all_summary_paths = []
    all_raw_paths = []

    for value in variation_values:
        if variation_mode == "hvg":
            n_hvg = int(value)
            n_pcs = int(fixed_pcs)
        else:
            n_hvg = int(fixed_hvg)
            n_pcs = int(value)

        cache_h5ad = hvg_cache_map[n_hvg]

        for variant in variants:
            logger.info("Running %s on %s | %s=%d", variant, dataset_name, variation_mode, value)

            if use_direct_camp4_mode and variant == "camp4":
                adata = load_recomb_processed_direct(
                    input_file=camp4_direct_input,
                    loom_var_name_key=loom_var_name_key,
                    n_pcs=n_pcs,
                )
                source_mode = "direct_recomb_processed"
            else:
                adata = load_cache_with_sliced_pca(cache_h5ad, n_pcs=n_pcs)
                source_mode = "cached_pipeline"

            prep_h5ad = os.path.join(
                prep_dir,
                f"{dataset_name}_{variant}_{variation_mode}_{value}_prep.h5ad"
            )
            if not os.path.exists(prep_h5ad):
                adata.write_h5ad(prep_h5ad, compression="gzip")

            prep_meta = os.path.join(
                prep_dir,
                f"{dataset_name}_{variant}_{variation_mode}_{value}_prep.meta.json"
            )
            write_metadata_json(prep_meta, {
                "dataset": dataset_name,
                "variant": variant,
                "variation_mode": variation_mode,
                "variation_value": int(value),
                "n_hvg": int(n_hvg),
                "n_pcs": int(n_pcs),
                "celltype_key": celltype_key,
                "gamma_list": [int(g) for g in gamma_list],
                "variants": variants,
                "source_mode": source_mode,
                "cache_input_file": input_file,
                "camp4_direct_input": camp4_direct_input,
            })

            part_csv = os.path.join(part_dir, f"{dataset_name}_{variant}_{variation_mode}_{value}_partitions.csv")
            run_meta = os.path.join(part_dir, f"{dataset_name}_{variant}_{variation_mode}_{value}_run.meta.json")

            if not os.path.exists(part_csv):
                if use_direct_camp4_mode and variant == "camp4":
                    logger.info("Running literal RECOMB CAMP4 exact mode")
                    np.random.seed(random_state)
                    run_camp4_recomb_exact(
                        adata=adata,
                        gamma_list=gamma_list,
                        output_csv=part_csv,
                        reduction_key="X_pca",
                        annotations=None,
                        min_metacells=1,
                    )
                else:
                    run_camp_variant(
                        adata=adata,
                        variant=variant,
                        gamma_list=gamma_list,
                        output_csv=part_csv,
                        annotations=None,
                        min_metacells=1,
                        random_state=random_state,
                    )

            write_metadata_json(run_meta, {
                "dataset": dataset_name,
                "variant": variant,
                "variation_mode": variation_mode,
                "variation_value": int(value),
                "n_hvg": int(n_hvg),
                "n_pcs": int(n_pcs),
                "partition_csv": part_csv,
                "prep_h5ad": prep_h5ad,
                "source_mode": source_mode,
                "celltype_key": celltype_key,
                "gamma_list": [int(g) for g in gamma_list],
                "cache_input_file": input_file,
                "camp4_direct_input": camp4_direct_input,
            })

            raw_metrics_csv = os.path.join(metrics_dir, f"{dataset_name}_{variant}_{variation_mode}_{value}_metrics_raw.csv")
            summary_metrics_csv = os.path.join(metrics_dir, f"{dataset_name}_{variant}_{variation_mode}_{value}_metrics_summary.csv")

            if not (os.path.exists(raw_metrics_csv) and os.path.exists(summary_metrics_csv)):
                compute_metrics_for_partition_file(
                    adata=adata,
                    partition_csv=part_csv,
                    dataset_name=dataset_name,
                    variant=variant,
                    variation_mode=variation_mode,
                    variation_value=value,
                    celltype_key=celltype_key,
                    out_raw_csv=raw_metrics_csv,
                    out_summary_csv=summary_metrics_csv,
                )

            all_raw_paths.append(raw_metrics_csv)
            all_summary_paths.append(summary_metrics_csv)

    summary_frames = [pd.read_csv(p) for p in all_summary_paths if os.path.exists(p)]
    raw_frames = [pd.read_csv(p) for p in all_raw_paths if os.path.exists(p)]

    if len(summary_frames) > 0:
        combined_summary = pd.concat(summary_frames, ignore_index=True)
        combined_summary_csv = os.path.join(metrics_dir, f"{dataset_name}_{variation_mode}_ALL_summary.csv")
        combined_summary.to_csv(combined_summary_csv, index=False)
        logger.info("Saved combined summary: %s", combined_summary_csv)
    else:
        combined_summary = pd.DataFrame()

    if len(raw_frames) > 0:
        combined_raw = pd.concat(raw_frames, ignore_index=True)
        combined_raw_csv = os.path.join(metrics_dir, f"{dataset_name}_{variation_mode}_ALL_raw.csv")
        combined_raw.to_csv(combined_raw_csv, index=False)
        logger.info("Saved combined raw metrics: %s", combined_raw_csv)
    else:
        combined_raw = pd.DataFrame()

    if not combined_summary.empty and not combined_raw.empty:
        make_all_plots_for_dataset_variation(
            summary_df=combined_summary,
            raw_df=combined_raw,
            out_dir=plots_dir,
            dataset_name=dataset_name,
            variation_mode=variation_mode,
            dpi=dpi,
            ecdf_gammas=ecdf_gammas,
        )


# =========================================================
# Argparse
# =========================================================
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_files", nargs="+", required=True)
    parser.add_argument("--dataset_names", nargs="+", required=True)
    parser.add_argument("--celltype_keys", nargs="+", required=True)

    parser.add_argument("--output_root", type=str, required=True)

    parser.add_argument("--variation_mode", type=str, choices=["hvg", "pca"], required=True)
    parser.add_argument("--variation_values", nargs="+", type=int, required=True)

    parser.add_argument("--fixed_hvg", type=int, default=2000)
    parser.add_argument("--fixed_pcs", type=int, default=50)

    parser.add_argument("--variants", nargs="+", default=["camp1", "camp2", "camp3", "camp4"])
    parser.add_argument("--gamma_list", nargs="+", type=int, default=[100, 150, 200, 250, 300, 350, 400, 450, 500, 550])

    parser.add_argument("--loom_var_name_key", type=str, default="gene_short_name")
    parser.add_argument("--min_genes", type=int, default=200)
    parser.add_argument("--min_cells", type=int, default=3)
    parser.add_argument("--random_state", type=int, default=0)
    parser.add_argument("--dpi", type=int, default=600)

    parser.add_argument("--ecdf_gammas", nargs="+", type=int, default=[50, 200, 500])

    # separate direct input for camp4
    parser.add_argument("--camp4_direct_input", type=str, default="")

    args = parser.parse_args()

    if not (len(args.input_files) == len(args.dataset_names) == len(args.celltype_keys)):
        raise ValueError("input_files, dataset_names, and celltype_keys must have the same length")

    return args


# =========================================================
# Main
# =========================================================
def main():
    args = parse_args()
    logger.info("Arguments:\n%s", json.dumps(vars(args), indent=2))

    for input_file, dataset_name, celltype_key in zip(args.input_files, args.dataset_names, args.celltype_keys):
        ckey = None if celltype_key == "NONE" else celltype_key

        one_dataset_one_variation(
            input_file=input_file,
            dataset_name=dataset_name,
            celltype_key=ckey,
            output_root=args.output_root,
            variation_mode=args.variation_mode,
            variation_values=args.variation_values,
            fixed_hvg=args.fixed_hvg,
            fixed_pcs=args.fixed_pcs,
            variants=args.variants,
            gamma_list=args.gamma_list,
            loom_var_name_key=args.loom_var_name_key,
            min_genes=args.min_genes,
            min_cells=args.min_cells,
            random_state=args.random_state,
            dpi=args.dpi,
            ecdf_gammas=args.ecdf_gammas,
            #camp4_direct_input=args.camp4_direct_input,
        )

    logger.info("All experiments finished.")


if __name__ == "__main__":
    main()
