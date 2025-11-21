import logging
#%%
import numpy as np
import pandas as pd
import scanpy as sc
import SEACells
import scanpy as sc
import scvelo as scv
import time
import sys
from sklearn.preprocessing import normalize


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
    
    logger.info("using rep_key of")
    logger.info(rep_key)

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
    #sample_scores = compute_lightweight_coreset_q(covid_b, rep_key="X_simi")
    sample_scores = compute_lightweight_coreset_q(covid_b, rep_key="X_pca")
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

def assign_adaptive_gaussian_to_seeds_on_the_fly(
    X_pca_all: np.ndarray,          # (N × d), float32
    subset_idx: np.ndarray,         # (n_subset,), global indices
    seed_idx: np.ndarray,           # (m,), global indices
    k_scale: int = 15,              # k for sigma estimation
    tau: float = 1.0,               # temperature
    bs: int = 20_000,               # batch size for subset rows
    eps: float = 1e-8,
):
    """
    One-shot assignment using an adaptive Gaussian computed only for (subset × seeds).
    Returns labels_idx (len = n_subset) in [0..m-1].
    """
    # extract embeddings
    Xs = X_pca_all[seed_idx]        # (m × d)
    m  = Xs.shape[0]
    k  = max(1, min(k_scale, m - 1))

    # precompute sigma for seeds (k-th neighbor among seeds)
    # pairwise distances among seeds (small: m×m)
    D_ss = pairwise_distances(Xs, Xs, metric='euclidean')   # (m × m)
    # sort each row, take the (k+1)-th smallest (skip self at 0)
    sigma_seed = np.partition(D_ss, kth=k, axis=1)[:, k].astype(np.float32)
    sigma_seed = np.maximum(sigma_seed, eps)

    labels_parts = []

    # process subset rows in batches
    for s in range(0, subset_idx.size, bs):
        e = min(s + bs, subset_idx.size)
        Xu = X_pca_all[subset_idx[s:e]]                    # (b × d)

        # distances cell↔seeds
        D_us = pairwise_distances(Xu, Xs, metric='euclidean')  # (b × m)

        # per-cell sigma_i = k-th neighbor among seeds for this cell
        # use partial sort to avoid full sort cost
        sigma_cell = np.partition(D_us, kth=k, axis=1)[:, k].astype(np.float32)
        sigma_cell = np.maximum(sigma_cell, eps)

        # adaptive Gaussian weights (b × m)
        denom = tau * (sigma_cell[:, None] * sigma_seed[None, :]) + eps
        # we already have Euclidean distances in D_us, so square them:
        W = np.exp(-(D_us ** 2) / denom, dtype=np.float32, where=~np.isinf(denom))

        # argmax over seeds
        labels_parts.append(np.argmax(W, axis=1))

        # free big blocks ASAP
        del D_us, sigma_cell, denom, W

    labels_idx = np.concatenate(labels_parts, axis=0)
    return labels_idx


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
                
                ## fast “nearest‐seed” via precomputed similarity
                #if reduction_key == 'X_simi':
                #    # global indices of this subset’s cells
                #    subset_idx = [adata.obs_names.get_loc(c) for c in subset.obs_names]
                #    # global indices of the coreset seeds
                #    seed_idx   = [adata.obs_names.get_loc(c) for c in coreset.obs_names]
                #    # slice the (n_subset × m) block of sim_matrix
                #    sim_sub    = adata.obsm['X_simi'][np.ix_(subset_idx, seed_idx)]
                #    # for each cell pick the seed with max similarity
                #    labels_idx = np.argmax(sim_sub, axis=1)
                #    labels     = [f'seed{lab}-{gamma}-{anno}' for lab in labels_idx]
             #   if reduction_key == 'X_simi':
             #       logger.info("X_simi: one-shot on sparse adaptive Gaussian + one cosine refinement")
             #       subset_idx = np.fromiter((adata.obs_names.get_loc(c) for c in subset.obs_names), dtype=np.int64)
             #       seed_idx   = np.fromiter((adata.obs_names.get_loc(c) for c in coreset.obs_names), dtype=np.int64)

             #       # 1) One-shot assignment from sparse K
             #       labels_idx = assign_from_sparse_K(adata, subset_idx, seed_idx)

             #       # 2) One refinement: centroids in X_unit + second-pass cosine assignment
             #       labels_idx = refine_once_cosine(adata, subset_idx, seed_idx, labels_idx, bs=100_000)

             #       labels = [f'seed{lab}-{gamma}-{anno}' for lab in labels_idx]
            
                if reduction_key == 'X_simi':
                    logger.info("X_simi: on-the-fly adaptive Gaussian to seeds + one cosine refinement")

                    subset_idx = np.fromiter((adata.obs_names.get_loc(c) for c in subset.obs_names), dtype=np.int64)
                    seed_idx   = np.fromiter((adata.obs_names.get_loc(c) for c in coreset.obs_names), dtype=np.int64)

                    X_pca_all = adata.obsm['X_pca'].astype(np.float32, copy=False)

                    # 1) one-shot adaptive Gaussian assignment to seeds (no global kernel)
                    t0 = time.time()
                    labels_idx = assign_adaptive_gaussian_to_seeds_on_the_fly(
                        X_pca_all,
                        subset_idx,
                        seed_idx,
                        k_scale=15,
                        tau=1.0,
                        bs=20_000,
                    )
                    logger.info("on-the-fly adaptive assign: %.2fs", time.time() - t0)

                    # 2) one-step refinement in cosine space
                    t1 = time.time()
                    labels_idx = refine_once_cosine(adata, subset_idx, seed_idx, labels_idx, bs=100_000)
                    logger.info("one-pass cosine refine: %.2fs", time.time() - t1)

                    labels = [f'seed{lab}-{gamma}-{anno}' for lab in labels_idx]


                else:
                    # fall back to PCA + KMeans
                    X_emb    = subset.obsm[reduction_key]
                    seed_idx = [subset.obs_names.get_loc(x) for x in coreset.obs_names]
                    seed_emb = X_emb[seed_idx]
                    km = KMeans(
                        n_clusters = m,
                        init       = seed_emb,
                        n_init     = 1,
                        max_iter   = 10,
                        algorithm  = 'lloyd',
                    ).fit(X_emb)
                    labels_idx = km.labels_
                    labels     = [f'seed{lab}-{gamma}-{anno}' for lab in labels_idx]

                
                # Assign via nearest-seed in PCA
                #X_emb    = subset.obsm[reduction_key]
                #seed_idx = [subset.obs_names.get_loc(x) for x in coreset.obs_names]
                #seed_emb = X_emb[seed_idx]
                ##labels_idx = pairwise_distances_argmin(X_emb, seed_emb)
                #km = KMeans(
                #    n_clusters = m,
                #    init       = seed_emb,
                #    n_init     = 1,
                #    max_iter   = 10,     # <-- a small number of refinement steps
                #    algorithm  = 'lloyd'
                #    ).fit(X_emb)
                #labels_idx = km.labels_
                #labels     = [f'seed{lab}-{gamma}-{anno}' for lab in labels_idx]
            cell_membership.loc[subset.obs_names, col] = labels
            seed_flag.loc[   seeds,         seed_col]   = True

        logger.info(f'Finished γ={gamma} (wrote {seed_flag[seed_col].sum()} seeds)')   
        current = time.time()
        used = current-start
        logger.info("used this many seconds for this gamma")
        logger.info(used)

    out = pd.concat([cell_membership, seed_flag], axis=1)
    out.to_csv(output_file)
    logger.info(f'Wrote membership+seed flags to {output_file}')
    #cell_membership.to_csv(output_file)
    #logger.info(f'Written results for gamma={gamma}')

    logger.info('All gammas processed')


def build_adaptive_gaussian_kernel(
    adata,
    use_rep='X_pca',
    n_neighbors=15,
    eps=1e-8,
    tau=1.0,              # temperature: <1.0 sharpens, >1.0 smooths
    symmetrize=True,
):
    """
    1) sc.pp.neighbors(... method='umap') builds sparse distances (approx kNN)
    2) Convert only those edges to adaptive Gaussian:
           K_ij = exp( - (d_ij^2) / (tau * sigma_i * sigma_j) )
       where sigma_i is i's kth-NN distance (row max in sparse distances).
    3) (Optional) symmetrize by max to ensure undirected similarity.
    """
    # neighbors populates adata.obsp['distances'] and ['connectivities']
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep=use_rep, method='umap', metric='euclidean')

    D = adata.obsp['distances'].tocsr().astype(np.float32, copy=False)  # (N×N) sparse, nnz ≈ N * n_neighbors
    sigma = np.maximum(D.max(axis=1).A.ravel(), eps)                    # per-row scale (kth NN distance)

    Dcoo = D.tocoo()
    den  = (sigma[Dcoo.row] * sigma[Dcoo.col]) * max(tau, eps) + eps
    sim  = np.exp(-(Dcoo.data**2) / den).astype(np.float32, copy=False)

    K = csr_matrix((sim, (Dcoo.row, Dcoo.col)), shape=D.shape)
    if symmetrize:
        K = K.maximum(K.T).tocsr()  # union via max
    adata.obsp['K_sim'] = K
    return K



def assign_from_sparse_K(adata, subset_idx, seed_idx):
    """
    Returns labels_idx in [0..m-1] via row-wise argmax over K_sim[subset, seeds].
    If a row has no edges to seeds, fallback uses cosine vs seeds in X_unit.
    """
    K = adata.obsp['K_sim'].tocsr()
    S = K[subset_idx][:, seed_idx]                  # (n_subset × m) csr

    indptr, indices, data = S.indptr, S.indices, S.data
    n_rows, _ = S.shape
    labels_idx = np.empty(n_rows, dtype=np.int64)
    empty_rows = []

    for i in range(n_rows):
        s, e = indptr[i], indptr[i+1]
        if e == s:
            empty_rows.append(i)
        else:
            labels_idx[i] = indices[s + np.argmax(data[s:e])]  # columns already 0..m-1

    if empty_rows:
        # tiny fallback: cosine on only these rows
        if 'X_unit' not in adata.obsm:
            adata.obsm['X_unit'] = normalize(adata.obsm['X_pca'], axis=1, copy=True)
        XU = adata.obsm['X_unit']
        empty_global = subset_idx[np.asarray(empty_rows, dtype=np.int64)]
        fill = (XU[empty_global] @ XU[seed_idx].T).argmax(axis=1)
        labels_idx[np.asarray(empty_rows, dtype=np.int64)] = fill

    return labels_idx


def refine_once_cosine(adata, subset_idx, seed_idx, labels_idx, bs=100_000):
    """
    Recompute seed centroids in X_unit, renormalize, then reassign via cosine.
    """
    from numpy.linalg import norm

    if 'X_unit' not in adata.obsm:
        adata.obsm['X_unit'] = normalize(adata.obsm['X_pca'], axis=1, copy=True)

    XU = adata.obsm['X_unit'].astype(np.float32, copy=False)
    m  = seed_idx.size

    # centroid recompute
    new_S = np.zeros((m, XU.shape[1]), dtype=np.float32)
    for k in range(m):
        members = (labels_idx == k)
        if np.any(members):
            c = XU[subset_idx[members]].mean(axis=0)
            nrm = float(norm(c))
            new_S[k] = c / (nrm if nrm > 0 else 1.0)
        else:
            new_S[k] = XU[seed_idx[k]]  # keep original seed if empty

    # second-pass assignment (batched to bound RAM)
    out = []
    for s in range(0, subset_idx.size, bs):
        e = min(s + bs, subset_idx.size)
        out.append((XU[subset_idx[s:e]] @ new_S.T).argmax(axis=1))
    return np.concatenate(out, axis=0)



if __name__ == '__main__':
    start = time.time()
    #input_file = '/scratch/dvl5760/blish_covid.seu.h5ad'
    input_file = "/storage/home/dvl5760/scratch/GSE156793_scjoint_subsample_prep.h5ad"
    adata = sc.read_h5ad(input_file)
    logger.info(f"done reading adata in {time.time()-start:.1f}s, shape: {adata.shape}")
    

    # ensure float32 PCA + unit-norm for cosine refinement
    adata.obsm['X_pca']  = adata.obsm['X_pca'].astype(np.float32, copy=False)
    adata.obsm['X_unit'] = normalize(adata.obsm['X_pca'], axis=1, copy=True)

    # fast adaptive Gaussian on existing kNN edges (stored as sparse pairwise matrix)
    #start = time.time()
    #build_adaptive_gaussian_kernel(
    #    adata,
    #    use_rep='X_pca',
    #    n_neighbors=15,
    #    tau=1.0,       # 0.7 sharper, 1.3 smoother
    #    symmetrize=True,
    #)
    #logger.info("built adaptive Gaussian kernel in %.2fs", time.time() - start)


#    start = time.time()
    covid_b = adata
#    #covid_b = adata[(adata.obs['Status'] == 'COVID') & (adata.obs["cell.type.coarse"]=="B")].copy()
#    #covid_b = adata[adata.obs['Status'] == 'COVID'].copy()
#    #covid_b = adata[(adata.obs['Status'] != 'COVID') & (adata.obs["cell.type.coarse"]=="B")].copy()
#    logger.info(f"selected COVIDâ€“B subset in {time.time()-start:.1f}s â†’ {covid_b.shape}")
#
#    logger.info("begin pre-processing covid_b")
#    sc.pp.filter_cells(covid_b, min_genes=200)
#    sc.pp.filter_genes(covid_b, min_cells=3)
#    logger.info("Shape after filtering: %s", covid_b.shape)
#
#
#    X_dense = covid_b.X.A if sparse.issparse(covid_b.X) else covid_b.X
#    num_neg = np.sum(X_dense < 0)
#    logger.warning("Number of negative entries in covid_b.X: %d", num_neg)
#
#    if num_neg > 0:
#        logger.warning("Clipping all negative values to 0")
#        covid_b.X = np.maximum(X_dense, 0)
#
#
#    total_counts = covid_b.X.sum(axis=1).A1 if sparse.issparse(covid_b.X) else covid_b.X.sum(axis=1)
#    logger.info("Min/Max total counts before normalization: %.1f / %.1f", total_counts.min(), total_counts.max())
#    logger.info("Number of cells with total count = 0: %d", np.sum(total_counts == 0))
#
#
#    sc.pp.normalize_total(covid_b, target_sum=1e4)
#    logger.info("Total-count normalization complete")
#    logger.info("Shape after normalization: %s", covid_b.shape)
#
#
#    sc.pp.log1p(covid_b)
#    logger.info("Log1p transformation complete")
#    logger.info("Shape after log1p: %s", covid_b.shape)
#
#    if not sparse.issparse(covid_b.X):
#        covid_b.X = csr_matrix(covid_b.X)
#        logger.info("Converted covid_b.X back to sparse matrix")
#
#    X_dense = covid_b.X.A if sparse.issparse(covid_b.X) else covid_b.X
#    gene_means = np.mean(X_dense, axis=0)
#    safe_genes_mask = ~np.isnan(gene_means) & ~np.isinf(gene_means)
#
#    if not np.all(safe_genes_mask):
#        logger.warning("Removing %d genes with NaN or Inf mean before HVG", (~safe_genes_mask).sum())
#        covid_b = covid_b[:, safe_genes_mask].copy()
#
#    if covid_b.shape[1] == 0:
#        logger.error("No genes left after cleaning. Exiting.")
#        exit(1)
#
#    sc.pp.highly_variable_genes(covid_b, n_top_genes=2000, flavor='seurat')
#    n_hvg = covid_b.var['highly_variable'].sum()
#    logger.info("Selected %d highly variable genes", n_hvg)
#    covid_b = covid_b[:, covid_b.var['highly_variable']].copy()
#    logger.info("Shape after hvg selection: %s", covid_b.shape)
#
#    sc.pp.scale(covid_b, max_value=10)
#    logger.info("Scaled gene expression")
#    sc.tl.pca(covid_b, svd_solver='arpack')
#    logger.info("Computed PCA")
#    logger.info("Shape after pca: %s", covid_b.shape)

    logger.info("done with pre-processing step")

    logger.info(f"covid with b cell: {covid_b.shape}")

    logger.info("about to run seacell + mcrigor on covid with b cell")

    #gamma_list = [20,30,32,35,38,42,47,53,60,70]
    #gamma_list = [20]
    #m_list, _ = create_m_list(covid_b, gamma_list)
    gamma_list = [100,150,200,250,300,350,400,450,500,550]

    #need to change covid_b into a similarity matrix which is derived from knn distance graph matrix
    #use adaptive gaussian kernel
#    from sklearn.neighbors import NearestNeighbors
#    from scipy.sparse import csr_matrix
#
#    def compute_adaptive_kernel(
#        X: np.ndarray,
#        n_neighbors: int = 15,
#        symmetrize: bool = True
#    ) -> csr_matrix:
#        """
#        Build a k-NN distance graph and convert it to an adaptive Gaussian kernel.
#
#        Parameters
#        ----------
#        X
#            Array of shape (n_cells, n_features), e.g. your PCA embedding.
#        n_neighbors
#            Number of neighbors for the k-NN graph.
#        symmetrize
#            If True, makes the graph undirected via union: D = max(D, D.T).
#
#        Returns
#        -------
#        K : csr_matrix
#            Sparse (n_cells × n_cells) adaptive Gaussian kernel matrix.
#        """
#        logger.info("Computing %d-NN distances", n_neighbors)
#        nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(X)
#        distances, indices = nbrs.kneighbors(X)
#
#        n_cells = X.shape[0]
#        # build sparse distance matrix D[i, j] = euclidean(X[i], X[j]) for j in NN(i)
#        rows = np.repeat(np.arange(n_cells), n_neighbors)
#        cols = indices.flatten()
#        data = distances.flatten()
#        D = csr_matrix((data, (rows, cols)), shape=(n_cells, n_cells))
#
#        if symmetrize:
#            logger.info("Symmetrizing k-NN graph (union)")
#            D = D.maximum(D.T)
#
#        # adaptive Gaussian kernel: sigma_i = distance to i's k-th neighbor
#        sigma = distances[:, -1]
#        logger.info("Building adaptive Gaussian kernel with per-cell sigma")
#
#        D_coo = D.tocoo()
#        sim = np.exp(
#            - (D_coo.data ** 2)
#            / (sigma[D_coo.row] * sigma[D_coo.col])
#        )
#
#        K = csr_matrix((sim, (D_coo.row, D_coo.col)), shape=D.shape)
#        logger.info("Kernel matrix constructed (nnz = %d)", K.nnz)
#        return K

    # build kernel on the PCA embedding
#    X_emb = covid_b.obsm['X_pca']
#    start = time.time()
#    K = compute_adaptive_kernel(X_emb, n_neighbors=15)
#    end = time.time()
#    logger.info("the seconds used to change covid_b into kernel matrix")
#    logger.info(end-start)
#    #done adding the changes
#
#    covid_b.obsm['X_simi'] = K

    m_list, _ = create_m_list(covid_b, gamma_list)

    SEACells_mcRigor(
        covid_b,
        #K,
        gamma_list,
        m_list,
        output_file = "/storage/home/dvl5760/work/SEACells/customized_metacell/human_fetal_atlas/output/edit_4_add_ad_gau_partitions.csv",
        #output_file='/storage/home/dvl5760/work/SEACells/customized_metacell/data/covid_healthy/edit_4_seacell_add_ad_gau_covid_b_partitions.csv',
        #reduction_key='X_pca'
        reduction_key='X_simi'
    )
    logger.info(f"Done in {time.time() - start:.1f}s")

