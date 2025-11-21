import logging
import time
import os

import numpy as np
import pandas as pd
import scanpy as sc

from scipy import sparse
from scipy.sparse import csr_matrix
from sklearn.metrics import pairwise_distances

# ----------------------------
# Logger
# ----------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.propagate = False

if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

logger.info('Start.')

# ----------------------------
# Load data
# ----------------------------
start = time.time()
#input_file = '/Users/danrongli/Desktop/yo/SEACells/data/blish_covid.seu.h5ad'
input_file = '/scratch/dvl5760/GSE156793_scjoint_subsample_prep.h5ad'
adata = sc.read_h5ad(input_file)
logger.info(f"Done reading adata in {time.time()-start:.1f}s, shape: {adata.shape}")

#start = time.time()
# If you truly want the full object, keep as-is. (Filter to subsets here if needed.)
covid_b = adata.copy()
#logger.info(f"Selected subset in {time.time()-start:.1f}s – shape: {covid_b.shape}")

# ----------------------------
# Pre-processing
# ----------------------------
#logger.info("Begin pre-processing covid_b")
#sc.pp.filter_cells(covid_b, min_genes=200)
#sc.pp.filter_genes(covid_b, min_cells=3)
#logger.info("Shape after filtering: %s", covid_b.shape)
#
## Clip negatives (just in case) and report totals
#X_dense_tmp = covid_b.X.A if sparse.issparse(covid_b.X) else covid_b.X
#num_neg = np.sum(X_dense_tmp < 0)
#logger.warning("Number of negative entries in covid_b.X: %d", num_neg)
#if num_neg > 0:
#    logger.warning("Clipping all negative values to 0")
#    covid_b.X = np.maximum(X_dense_tmp, 0)
#else:
#    # ensure we keep original sparsity when possible
#    if not sparse.issparse(covid_b.X):
#        covid_b.X = csr_matrix(covid_b.X)
#
#total_counts = covid_b.X.sum(axis=1).A1 if sparse.issparse(covid_b.X) else covid_b.X.sum(axis=1)
#logger.info("Min/Max total counts before normalization: %.1f / %.1f", float(total_counts.min()), float(total_counts.max()))
#logger.info("Number of cells with total count = 0: %d", int(np.sum(total_counts == 0)))
#
#sc.pp.normalize_total(covid_b, target_sum=1e4)
#sc.pp.log1p(covid_b)
#
## Ensure sparse (saves RAM downstream)
#if not sparse.issparse(covid_b.X):
#    covid_b.X = csr_matrix(covid_b.X)
#
## Drop genes with NaN/Inf stats (defensive)
#X_dense_check = covid_b.X.A if sparse.issparse(covid_b.X) else covid_b.X
#gene_means = np.mean(X_dense_check, axis=0)
#safe_genes_mask = ~np.isnan(gene_means) & ~np.isinf(gene_means)
#covid_b = covid_b[:, safe_genes_mask].copy()
#
## HVGs, scale, PCA
#sc.pp.highly_variable_genes(covid_b, n_top_genes=2000, flavor='seurat')
#covid_b = covid_b[:, covid_b.var['highly_variable']].copy()
#sc.pp.scale(covid_b, max_value=10)
#sc.tl.pca(covid_b, svd_solver='arpack')
#logger.info("Done with pre-processing.")

# Neighborhood graph + diffusion map (for centroid separation in diffusion space)
sc.pp.neighbors(covid_b, use_rep='X_pca')
sc.tl.diffmap(covid_b)
X_diffmap = covid_b.obsm['X_diffmap']            # (n_cells, n_diffmap_components)
X_diff_dense = X_diffmap if not sparse.issparse(X_diffmap) else X_diffmap.toarray()
logger.info("Computed diffusion space.")

# Also keep a dense view of expression for INV metric later
#X_expr = covid_b.X.A if sparse.issparse(covid_b.X) else covid_b.X
X_expr = covid_b.X  # keep sparse (CSR)

# ----------------------------
# Load metacell partitions and align cells
# ----------------------------
#metacell_partition = pd.read_csv(
#    "/Users/danrongli/Desktop/yo/partition_summary/covid_healthy/adding_seed_umap_simi/edit_4_seacell_covid_b_partitions.csv",
#    index_col=0
#)

#metacell_partition = pd.read_csv(
#    "/storage/home/dvl5760/work/SEACells/customized_metacell/human_fetal_atlas/output/edit_4_partitions.csv",
#    index_col=0
#)
metacell_partition = pd.read_csv(
    "/storage/home/dvl5760/work/SEACells/customized_metacell/human_fetal_atlas/output/edit_4_partitions_large_metacell.csv",
    index_col=0
)

assert X_diff_dense.shape[0] == metacell_partition.shape[0], \
    "Mismatch between number of cells and partition rows"

#common_cells = covid_b.obs.index.intersection(metacell_partition.index)
#covid_b = covid_b[common_cells].copy()
#metacell_partition = metacell_partition.loc[common_cells]
#X_diff_dense = X_diff_dense[covid_b.obs.index.get_indexer(common_cells)]
#X_expr       = X_expr[covid_b.obs.index.get_indexer(common_cells)]


# ----------------------------
# Which gamma columns (partition resolutions) to evaluate
# ----------------------------
#m_list = [4945, 3297, 2472, 1978, 1648, 1413, 1236, 1099, 989, 899]
m_list = [24729	,16486,12364,9891,8243,7065]
g_list = metacell_partition.columns[0:len(m_list)]

# ----------------------------
# Metrics helpers
# ----------------------------
def compute_purity(labels_for_group, full_labels_series, label_col="Main_cluster_name"):
    """Majority fraction within a metacell."""
    subset = full_labels_series.loc[labels_for_group.index]
    counts = subset.value_counts()
    return float(counts.max()) / float(len(subset)) if len(subset) > 0 else np.nan

def compute_compactness(diff_coords):
    """Larger is better: we use negative mean variance across diffusion dims."""
    return float(-np.var(diff_coords, axis=0).mean())

def compute_INV(expr_block):
    """
    95th percentile of gene-level (variance / mean) inside the metacell.
    Adds 1e-8 to denominator for stability.
    """
    if sparse.issparse(expr_block):
        expr_block = expr_block.toarray()
    gene_var = np.var(expr_block, axis=0)
    gene_mean = np.mean(expr_block, axis=0)
    inv = np.percentile(gene_var / (gene_mean + 1e-8), 95)
    return float(inv)

# ----------------------------
# Main loop
# ----------------------------
results = []

for i in range(len(m_list)):
    gamma_col = g_list[i]
    m_col = m_list[i]
    logger.info(f"Evaluating partition {gamma_col} with m={m_col}")

    #covid_b.obs["metacell"] = metacell_partition[gamma_col]
    covid_b.obs["metacell"] = metacell_partition[gamma_col].to_numpy()  # or .values

    col = gamma_col
    ser = metacell_partition[col]

    assert len(ser) == covid_b.n_obs, "Row count mismatch"

    covid_b.obs["metacell"] = ser.to_numpy()
    n_nan = covid_b.obs["metacell"].isna().sum()
    logger.info(f"Non-null metacell labels: {covid_b.n_obs - n_nan} / {covid_b.n_obs}")
    assert n_nan == 0, "All metacell labels became NaN (index alignment issue)"
    logger.info(f"Unique metacells: {covid_b.obs['metacell'].nunique()}")
    
    labels = covid_b.obs["metacell"].values

    # Per-metacell metrics
    metrics = []
    # Group once for efficiency
    grouped = covid_b.obs.groupby("metacell")

    # Precompute centroids in diffusion space per metacell
    centroids = {}
    for metacell_id, group in grouped:
        idx = group.index
        ix = covid_b.obs.index.get_indexer(idx)
        diff_coords = X_diff_dense[ix]
        centroids[metacell_id] = diff_coords.mean(axis=0)

    # Compute separation = nearest-centroid distance in diffusion space
    # (order aligned to iteration below)
    mc_ids = list(centroids.keys())
    C = np.vstack([centroids[mid] for mid in mc_ids])
    D = pairwise_distances(C, C)
    np.fill_diagonal(D, np.inf)
    nearest_sep = np.min(D, axis=1)
    sep_map = {mid: float(nearest_sep[j]) for j, mid in enumerate(mc_ids)}

    # Now compute purity, compactness, INV
    for metacell_id, group in grouped:
        idx = group.index
        ix = covid_b.obs.index.get_indexer(idx)
        purity_val = compute_purity(group, covid_b.obs["Main_cluster_name"])

        # Compactness in diffusion space
        diff_coords = X_diff_dense[ix]
        compact_val = compute_compactness(diff_coords)

        # INV from expression inside the group
        expr_block = X_expr[ix, :]
        inv_val = compute_INV(expr_block)

        # Append only requested fields
        metrics.append((
            metacell_id,
            purity_val,
            compact_val,
            sep_map[metacell_id],
            inv_val
        ))

    metacell_ids, purities, compactnesses, separations, invs = zip(*metrics)

    df = pd.DataFrame({
        "m": m_col,
        "metacell_id": metacell_ids,
        "purity": purities,
        "compactness": compactnesses,
        "separation": separations,
        "INV": invs
    })

    results.append(df)

# ----------------------------
# Save outputs
# ----------------------------
final_df = pd.concat(results, ignore_index=True)

#out_dir = "/Users/danrongli/Desktop/yo/partition_summary/covid_healthy/metrics/update_version/adding_seed_umap_simi"
out_dir = "/storage/home/dvl5760/work/SEACells/metrics"
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "camp1_metrics_human_fetal_atlas_large_metacell.csv")

final_df.to_csv(out_path, index=False)
logger.info(f"Saved metrics to: {out_path}")
logger.info("Done.")
