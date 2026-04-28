import logging,sys
#%%
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import scanpy as sc
import SEACells
import scanpy as sc
import scvelo as scv
import time

from anndata import AnnData
from typing import Optional
from scipy import sparse
from scipy.sparse import csr_matrix
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin

# Logger setup
#def setup_logger():
#    logger = logging.getLogger(__name__)
#    logger.setLevel(logging.INFO)
#    if not logger.handlers:
#        ch = logging.StreamHandler()
#        ch.setLevel(logging.INFO)
#        fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#        ch.setFormatter(fmt)
#        logger.addHandler(ch)
#    return logger

#logger = setup_logger()
#logger.info('done importing stuff')
print("done importing stuff")


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,   # ensures duplicate handlers are cleared
)

logger = logging.getLogger(__name__)
logger.info("Logger ready")




def compute_lightweight_coreset_q(adata: AnnData, rep_key: Optional[str] = None, layer: Optional[str] = None) -> np.ndarray:
    """
    Compute q(x) for each observation in PCA space (if rep_key) or raw counts.
    """
    if rep_key is not None:
        X = adata.obsm[rep_key]
    elif layer is not None:
        X = adata.layers[layer]
    else:
        X = adata.X

    if sparse.issparse(X):
        logger.info("X is sparse format")
        X_csr = X.tocsr(copy=False)
        mu = np.asarray(X_csr.mean(axis=0)).ravel()
        x_norm_sq = np.asarray(X_csr.power(2).sum(axis=1)).ravel()
        mu_norm_sq = np.dot(mu, mu)
        prod = X_csr @ mu
        d_sq = x_norm_sq + mu_norm_sq - 2.0 * prod
    else:
        logger.info("X is dense format")
        X = np.asarray(X, dtype=np.float64)
        mu = X.mean(axis=0)
        d_sq = np.sum((X - mu) ** 2, axis=1)

    n = adata.n_obs
    denom = d_sq.sum()
    if denom == 0.0:
        logger.warning("All points coincide; returning uniform distribution.")
        return np.full(n, 1.0 / n)

    q = 0.5 * (1.0 / n) + 0.5 * (d_sq / denom)
    q /= q.sum()
    return q


def create_m_list(covid_b: AnnData, gamma_list: list) -> (list, np.ndarray):
    start = time.time()
    # compute q in PCA space
    sample_scores = compute_lightweight_coreset_q(covid_b, rep_key='X_pca')
    covid_b.obs['q'] = sample_scores
    logger.info("computed q(x) in %.1fs (sanity: q.sum() = %.4f)", time.time() - start, sample_scores.sum())

    n_cells = covid_b.n_obs
    m_list = [int(n_cells / g) for g in gamma_list]
    return m_list, sample_scores


def create_coreset_unselected(covid_b: AnnData, m: int, g: float, sample_scores: np.ndarray):
    logger.info(f"current m is {m}, gamma is {g}")
    all_idx = np.arange(covid_b.n_obs)
    idx = np.random.choice(n_cells := covid_b.n_obs, size=m, replace=False, p=sample_scores)
    coreset    = covid_b[idx].copy()
    unselected = covid_b[np.setdiff1d(all_idx, idx)].copy()
    logger.info("coreset created: %s", coreset.shape)
    return coreset, unselected


def SEACells_mcRigor(
    adata: AnnData,
    gamma_list: list,
    m_list: list,
    output_file: str = 'seacells_membership.csv',
    reduction_key: str = 'X_pca',
    annotations: Optional[str] = None,
    min_metacells: int = 1
):
    if annotations is None or annotations not in adata.obs:
        annotations = 'SEACell_batch'
        adata.obs[annotations] = 'allcells'

    cell_membership = pd.DataFrame(index=adata.obs_names)
    seed_flag       = pd.DataFrame(index=adata.obs_names)

    for gamma, m in zip(gamma_list, m_list):
        col = str(gamma)
        seed_col = f'{col}_is_seed'

        cell_membership[col] = None
        seed_flag[seed_col]  = False

        logger.info(f'Gamma={gamma}, m={m}')
        start = time.time()
        for anno in adata.obs[annotations].unique():
            subset = adata[adata.obs[annotations] == anno].copy()
            if subset.n_obs <= min_metacells:
                labels = [f'{anno}-SEACell-1'] * subset.n_obs
                seeds  = []   # no actual sampling
            else:
                scores = subset.obs['q'].values if 'q' in subset.obs else compute_lightweight_coreset_q(subset, rep_key=reduction_key)
                subset.obs['q'] = scores
                coreset, _ = create_coreset_unselected(subset, m, gamma, scores)
                seeds = list(coreset.obs_names)

                # Assign via nearest-seed in PCA
                X_emb    = subset.obsm[reduction_key]
                seed_idx = [subset.obs_names.get_loc(x) for x in coreset.obs_names]
                seed_emb = X_emb[seed_idx]
                #labels_idx = pairwise_distances_argmin(X_emb, seed_emb)
                km = KMeans(
                    n_clusters = m,
                    init       = seed_emb,
                    n_init     = 1,
                    max_iter   = 10,     # <-- a small number of refinement steps
                    algorithm  = 'lloyd'
                    ).fit(X_emb)
                labels_idx = km.labels_
                labels     = [f'seed{lab}-{gamma}-{anno}' for lab in labels_idx]
            cell_membership.loc[subset.obs_names, col] = labels
            seed_flag.loc[   seeds,         seed_col]   = True
        
        logger.info(f'Finished γ={gamma} (wrote {seed_flag[seed_col].sum()} seeds)')
        current =  time.time()
        used = current-start
        logger.info("used this many seconds for this gamma")
        logger.info(used)

    out = pd.concat([cell_membership, seed_flag], axis=1)
    out.to_csv(output_file)
    logger.info(f'Wrote membership+seed flags to {output_file}')
    #cell_membership.to_csv(output_file)
    #logger.info(f'Written results for gamma={gamma}')

    logger.info('All gammas processed')


if __name__ == '__main__':
    logger.info("Quick loom check: opening file")
    import loompy
    with loompy.connect("/storage/home/dvl5760/scratch/GSE156793_scjoint_subsample.loom", "r") as ds:
        logger.info("shape: %s", ds.shape)
        logger.info("row attrs (sample): %s", list(ds.ra)[:5])
        logger.info("col attrs (sample): %s", list(ds.ca)[:5])
    logger.info("Quick loom check done")


    start = time.time()
    #input_file = '/scratch/dvl5760/blish_covid.seu.h5ad'
    #adata = sc.read_h5ad(input_file)
    input_file = '/storage/home/dvl5760/scratch/GSE156793_scjoint_subsample.loom'
    adata = sc.read_loom(input_file, sparse=True, dtype='float32')
    logger.info(f"done reading adata in {time.time()-start:.1f}s, shape: {adata.shape}")
    
    # --- Set gene symbols as variable names ---
    if 'gene_short_name' in adata.var.columns:
        adata.var_names = adata.var['gene_short_name']
        logger.info("Set adata.var_names from gene_short_name.")
    else:
        logger.warning("gene_short_name not found; keeping default var_names.")

    start = time.time()
    covid_b = adata
    #covid_b = adata[(adata.obs['Status'] == 'COVID') & (adata.obs["cell.type.coarse"]=="B")].copy()
    #covid_b = adata[adata.obs['Status'] == 'COVID'].copy()
    #covid_b = adata[(adata.obs['Status'] != 'COVID') & (adata.obs["cell.type.coarse"]=="B")].copy()
    logger.info(f"selected COVIDâ€“B subset in {time.time()-start:.1f}s â†’ {covid_b.shape}")

    logger.info("begin pre-processing covid_b")
    sc.pp.filter_cells(covid_b, min_genes=200)
    sc.pp.filter_genes(covid_b, min_cells=3)
    logger.info("Shape after filtering: %s", covid_b.shape)

    #remove the densifying part since we are reading loom file
    #X_dense = covid_b.X.A if sparse.issparse(covid_b.X) else covid_b.X
    #num_neg = np.sum(X_dense < 0)
    #logger.warning("Number of negative entries in covid_b.X: %d", num_neg)
    #
    #if num_neg > 0:
    #    logger.warning("Clipping all negative values to 0")
    #    covid_b.X = np.maximum(X_dense, 0)
    #
    #
    #total_counts = covid_b.X.sum(axis=1).A1 if sparse.issparse(covid_b.X) else covid_b.X.sum(axis=1)
    #logger.info("Min/Max total counts before normalization: %.1f / %.1f", total_counts.min(), total_counts.max())
    #logger.info("Number of cells with total count = 0: %d", np.sum(total_counts == 0))
    
    #add the correct sparse ones for loom

    # --- nonnegative check without densifying ---
    if sparse.issparse(covid_b.X):
        num_neg = int((covid_b.X.data < 0).sum())
    else:
        num_neg = int((covid_b.X < 0).sum())

    logger.warning("Number of negative entries in covid_b.X: %d", num_neg)
    if num_neg > 0:
        logger.warning("Clipping all negative values to 0")
        if sparse.issparse(covid_b.X):
            covid_b.X.data = np.maximum(covid_b.X.data, 0.0)
        else:
            covid_b.X = np.maximum(covid_b.X, 0.0)

    # --- totals without densifying ---
    if sparse.issparse(covid_b.X):
        total_counts = np.ravel(covid_b.X.sum(axis=1))
    else:
        total_counts = covid_b.X.sum(axis=1)

    logger.info("Min/Max total counts before normalization: %.1f / %.1f", float(total_counts.min()), float(total_counts.max()))
    logger.info("Number of cells with total count = 0: %d", int(np.sum(total_counts == 0)))

    #done adding

    sc.pp.normalize_total(covid_b, target_sum=1e4)
    logger.info("Total-count normalization complete")
    logger.info("Shape after normalization: %s", covid_b.shape)


    sc.pp.log1p(covid_b)
    logger.info("Log1p transformation complete")
    logger.info("Shape after log1p: %s", covid_b.shape)
    
    #removing dense again
    #if not sparse.issparse(covid_b.X):
    #    covid_b.X = csr_matrix(covid_b.X)
    #    logger.info("Converted covid_b.X back to sparse matrix")
    #
    #X_dense = covid_b.X.A if sparse.issparse(covid_b.X) else covid_b.X
    #gene_means = np.mean(X_dense, axis=0)
    #safe_genes_mask = ~np.isnan(gene_means) & ~np.isinf(gene_means)
    #
    #if not np.all(safe_genes_mask):
    #    logger.warning("Removing %d genes with NaN or Inf mean before HVG", (~safe_genes_mask).sum())
    #    covid_b = covid_b[:, safe_genes_mask].copy()
    #
    #if covid_b.shape[1] == 0:
    #    logger.error("No genes left after cleaning. Exiting.")
    #    exit(1)
    
    # ensure CSR for fast ops
    if sparse.issparse(covid_b.X):
        covid_b.X = covid_b.X.tocsr()

    # --- robust gene mean check without densifying ---
    gene_means = np.asarray(covid_b.X.mean(axis=0)).ravel() if sparse.issparse(covid_b.X) else covid_b.X.mean(axis=0)
    safe_genes_mask = np.isfinite(gene_means)

    if not np.all(safe_genes_mask):
        logger.warning("Removing %d genes with NaN/Inf mean before HVG", int((~safe_genes_mask).sum()))
        covid_b = covid_b[:, safe_genes_mask].copy()

    if covid_b.shape[1] == 0:
        logger.error("No genes left after cleaning. Exiting.")
        raise SystemExit(1)
    #done adding

    sc.pp.highly_variable_genes(covid_b, n_top_genes=2000, flavor='seurat')
    n_hvg = covid_b.var['highly_variable'].sum()
    logger.info("Selected %d highly variable genes", n_hvg)
    covid_b = covid_b[:, covid_b.var['highly_variable']].copy()
    logger.info("Shape after hvg selection: %s", covid_b.shape)
    
    #remove densing part again
    #sc.pp.scale(covid_b, max_value=10)
    #logger.info("Scaled gene expression")
    #sc.tl.pca(covid_b, svd_solver='arpack')
    #logger.info("Computed PCA")
    #logger.info("Shape after pca: %s", covid_b.shape)
    
    sc.tl.pca(covid_b, svd_solver='arpack', zero_center=False, n_comps=50)
    logger.info("Computed PCA (zero_center=False, sparse-friendly). X_pca shape: %s", covid_b.obsm['X_pca'].shape)
    #done adding

    logger.info("done with pre-processing step")

    logger.info(f"covid with b cell: {covid_b.shape}")
    

    if 'gene_short_name' in covid_b.var.columns:
        covid_b.var['gene_short_name'] = covid_b.var['gene_short_name'].astype(str)
        covid_b.var_names = covid_b.var['gene_short_name']
        covid_b.var_names_make_unique()
        covid_b.var.drop(columns=['gene_short_name'], inplace=True)
        covid_b.var.index.name = None
    else:
        covid_b.var.index.name = None

    covid_b.write_h5ad('/storage/home/dvl5760/scratch/GSE156793_scjoint_subsample_prep.h5ad', compression='gzip')
    logger.info("saved the pre-processed data into scratch")

    logger.info("about to run seacell + mcrigor on covid with b cell")

    #gamma_list = [20,30,32,35,38,42,47,53,60,70] #covid_healthy 44k
    gamma_list = [100,150,200,250,300,350,400,450,500,550]

    m_list, _ = create_m_list(covid_b, gamma_list)

    SEACells_mcRigor(
        covid_b,
        gamma_list,
        m_list,
        #output_file='/storage/home/dvl5760/work/SEACells/customized_metacell/data/covid_healthy/edit_4_seacell_covid_b_partitions.csv',
        output_file = "/storage/home/dvl5760/work/SEACells/customized_metacell/human_fetal_atlas/output/edit_4_partitions.csv",
        reduction_key='X_pca'
    )
    logger.info(f"Done in {time.time() - start:.1f}s")
    

    import sys, logging, time, threading, os

    logger.info("All work complete. Beginning clean shutdown.")
    logger.info("Live threads at end: %s", [(t.name, t.daemon) for t in threading.enumerate()])

    # optional tiny pause
    time.sleep(0.2)

    # flush visible output before killing the process
    sys.stdout.flush(); sys.stderr.flush()

    # close logging handlers
    logging.shutdown()

    # hard-terminate process (no atexit handlers, no lingering threads)
    os._exit(0)
