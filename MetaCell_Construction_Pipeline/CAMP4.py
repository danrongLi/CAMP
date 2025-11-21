import logging, sys, os, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import scanpy as sc
import SEACells
from SEACells import build_graph
from anndata import AnnData
from typing import Optional, List, Tuple
from scipy import sparse
from scipy.sparse import csr_matrix

# =========================================================
# logging
# =========================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,   # clear old handlers
)
logger = logging.getLogger(__name__)
logger.info("Logger ready")


# =========================================================
# 1. q(x) – importance for coreset
# =========================================================
def compute_lightweight_coreset_q(
    adata: AnnData,
    rep_key: Optional[str] = None,
    layer: Optional[str] = None
) -> np.ndarray:
    """
    q(x) = 0.5 * uniform + 0.5 * (dist-to-mean / sum-dist).
    This is just to bias sampling toward “spread-out” cells.
    """
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
        logger.warning("All points coincide; using uniform q.")
        return np.full(n, 1.0 / n)

    q = 0.5 * (1.0 / n) + 0.5 * (d_sq / denom)
    q /= q.sum()
    return q


# =========================================================
# 2. m_list from gamma_list
# =========================================================
def create_m_list(adata: AnnData, gamma_list: List[int]) -> Tuple[List[int], np.ndarray]:
    start = time.time()
    q = compute_lightweight_coreset_q(adata, rep_key="X_pca")
    adata.obs["q"] = q
    logger.info("computed q(x) in %.1fs; q.sum()=%.4f", time.time() - start, q.sum())
    n_cells = adata.n_obs
    m_list = [max(1, int(n_cells / g)) for g in gamma_list]
    return m_list, q


# =========================================================
# 3. sample coreset = FINAL archetypes
# =========================================================
def sample_coreset(subset: AnnData, m: int, probs: np.ndarray) -> AnnData:
    """
    Choose m cells from subset using probs → this *is* the final set of archetypes.
    """
    n_sub = subset.n_obs
    idx = np.random.choice(n_sub, size=m, replace=False, p=probs)
    return subset[idx].copy()


# =========================================================
# 4. SEACells-style assignment from fixed archetypes
#    A = K @ B, argmax, no alternating
# =========================================================

def assign_from_fixed_archetypes(
    subset: AnnData,
    seed_names: List[str],
    reduction_key: str = "X_pca",
    n_neighbors: int = 30,
    graph_construction: str = "union",
    chunk_size: int = 20000,
) -> np.ndarray:
    """
    subset: AnnData for this batch
    seed_names: obs_names of cells that are the final archetypes
    return: array of length n_sub with cluster indices (0..m-1)
    """
    # 1) build SAME kernel as SEACells (robust to version differences)
    g = build_graph.SEACellGraph(subset, reduction_key, verbose=False)

    try:
        # newer API
        K = g.rbf(n_neighbors=n_neighbors, graph_construction=graph_construction)
    except TypeError:
        # older API: n_neighbors already stored inside the object
        g.n_neighbors = n_neighbors
        K = g.rbf(graph_construction=graph_construction)

    K = K.tocsr()
    n_sub = subset.n_obs

    # 2) build B = one-hot on seeds
    m_seeds = len(seed_names)
    seed_idx = [subset.obs_names.get_loc(s) for s in seed_names]
    data = np.ones(m_seeds, dtype=np.float32)
    rows = np.array(seed_idx, dtype=int)
    cols = np.arange(m_seeds, dtype=int)
    B = csr_matrix((data, (rows, cols)), shape=(n_sub, m_seeds))

    # 3) multiply in chunks to avoid OOM
    labels = np.empty(n_sub, dtype=int)
    for start in range(0, n_sub, chunk_size):
        end = min(start + chunk_size, n_sub)
        K_block = K[start:end, :]        # (block x n_sub)
        A_block = K_block.dot(B)         # (block x m_seeds)
        A_block = A_block.toarray()
        labels[start:end] = A_block.argmax(axis=1)

    return labels




def assign_from_fixed_archetypes_previous(
    subset: AnnData,
    seed_names: List[str],
    reduction_key: str = "X_pca",
    n_neighbors: int = 30,
    graph_construction: str = "union",
    chunk_size: int = 20000,
) -> np.ndarray:
    """
    subset: AnnData for this batch
    seed_names: obs_names of cells that are the final archetypes
    return: array of length n_sub with cluster indices (0..m-1)
    """
    # 1) build SAME kernel as SEACells
    g = build_graph.SEACellGraph(subset, reduction_key, verbose=False)
    K = g.rbf(n_neighbors=n_neighbors, graph_construction=graph_construction)
    K = K.tocsr()
    n_sub = subset.n_obs

    # 2) build B = one-hot on seeds
    m_seeds = len(seed_names)
    seed_idx = [subset.obs_names.get_loc(s) for s in seed_names]
    data = np.ones(m_seeds, dtype=np.float32)
    rows = np.array(seed_idx, dtype=int)
    cols = np.arange(m_seeds, dtype=int)
    B = csr_matrix((data, (rows, cols)), shape=(n_sub, m_seeds))

    # 3) multiply in chunks to avoid OOM
    labels = np.empty(n_sub, dtype=int)
    for start in range(0, n_sub, chunk_size):
        end = min(start + chunk_size, n_sub)
        K_block = K[start:end, :]        # (block x n_sub)
        A_block = K_block.dot(B)         # (block x m_seeds)
        A_block = A_block.toarray()
        labels[start:end] = A_block.argmax(axis=1)

    return labels


# =========================================================
# 5. main runner
# =========================================================
def SEACells_mcRigor_fixed(
    adata: AnnData,
    gamma_list: List[int],
    m_list: List[int],
    output_file: str,
    reduction_key: str = "X_pca",
    annotations: Optional[str] = None,
    min_metacells: int = 1,
):
    """
    For each gamma:
      - m = n/gamma
      - for each batch:
          1. sample coreset (THIS is the final archetype set)
          2. build SEACells kernel
          3. assign via K@B, argmax
      - save
    No SEACells alternating, no cpu.py initialize(), no fit().
    """
    if annotations is None or annotations not in adata.obs:
        annotations = "SEACell_batch"
        adata.obs[annotations] = "allcells"

    cell_membership = pd.DataFrame(index=adata.obs_names)
    seed_flag       = pd.DataFrame(index=adata.obs_names)

    for gamma, m in zip(gamma_list, m_list):
        col = str(gamma)
        seed_col = f"{col}_is_seed"
        cell_membership[col] = None
        seed_flag[seed_col]  = False

        logger.info("=== gamma=%s, m≈%s ===", gamma, m)
        t_gamma = time.time()

        for anno in adata.obs[annotations].unique():
            subset = adata[adata.obs[annotations] == anno].copy()

            # make sure we have PCA
            if reduction_key not in subset.obsm:
                sc.tl.pca(subset, n_comps=50, zero_center=False)

            # very small group
            if subset.n_obs <= min_metacells:
                labels = [f"{anno}-SEACell-1"] * subset.n_obs
                cell_membership.loc[subset.obs_names, col] = labels
                continue

            # probs inside this subset
            if "q" in subset.obs:
                q_sub = subset.obs["q"].to_numpy()
            else:
                q_sub = compute_lightweight_coreset_q(subset, rep_key=reduction_key)
                subset.obs["q"] = q_sub

            # 1) sample coreset = FINAL archetypes
            m_here = min(m, subset.n_obs)  # cannot sample more seeds than cells
            coreset = sample_coreset(subset, m_here, q_sub)
            seed_names = list(coreset.obs_names)
            logger.info("%s: sampled %d seeds (final archetypes)", anno, len(seed_names))

            # 2) assign in kernel space (SEACells style), no alternating
            labels_idx = assign_from_fixed_archetypes(
                subset,
                seed_names,
                reduction_key=reduction_key,
                n_neighbors=30,
                graph_construction="union",
                chunk_size=20000,
            )

            # 3) pretty labels
            final_labels = [f"seed{lab}-{gamma}-{anno}" for lab in labels_idx]

            # write back
            cell_membership.loc[subset.obs_names, col] = final_labels
            seed_flag.loc[seed_names, seed_col] = True

            logger.info("%s: done (%d cells)", anno, subset.n_obs)

        logger.info("gamma=%s done in %.1fs", gamma, time.time() - t_gamma)

    out = pd.concat([cell_membership, seed_flag], axis=1)
    out.to_csv(output_file)
    logger.info("Wrote %s", output_file)
    logger.info("All gammas processed")


# =========================================================
# 6. main
# =========================================================
if __name__ == "__main__":
    start = time.time()

    input_file = "/storage/home/dvl5760/scratch/GSE156793_scjoint_subsample_prep.h5ad"
    adata = sc.read_h5ad(input_file)
    logger.info("read %s in %.1fs", adata.shape, time.time() - start)

    # make PCA once
    if "X_pca" not in adata.obsm:
        sc.tl.pca(adata, n_comps=50, zero_center=False)
        logger.info("computed PCA; X_pca shape = %s", adata.obsm["X_pca"].shape)

    # your gammas
    gamma_list = [20, 30, 40, 50, 60, 70, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550]
    m_list, _ = create_m_list(adata, gamma_list)

    out_csv = "/storage/home/dvl5760/work/SEACells/customized_metacell/human_fetal_atlas/output/edit_5_partitions_full_metacell.csv"
    SEACells_mcRigor_fixed(
        adata,
        gamma_list,
        m_list,
        output_file=out_csv,
        reduction_key="X_pca",
        annotations=None,
        min_metacells=1,
    )

    logger.info("ALL DONE in %.1fs", time.time() - start)
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(0)
