import logging
from multiprocessing import Pool
import cProfile, pstats, io
#%%
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

os.environ["TQDM_DISABLE"]="0"
import numpy as np
import pandas as pd
import scanpy as sc
import SEACells
import scanpy as sc
import scvelo as scv
import time

import scipy.sparse as sp
from anndata import AnnData
from typing import Optional
from scipy import sparse
from scipy.sparse import csr_matrix
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cdist


# Create or get the logger
logger = logging.getLogger(__name__)

# Set the level of the logger. This is optional and can be set to other levels (e.g., DEBUG, ERROR)
logger.setLevel(logging.INFO)

# Create a console handler and set the level to INFO
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Create a formatter and set the formatter for the handler
#formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - PID:%(process)d - Thread:%(threadName)s - %(message)s'
)
console_handler.setFormatter(formatter)

# Add the handler to the logger
logger.addHandler(console_handler)

logger.info('done importing stuff')


def _run_gamma(args):
    gamma, adata, annotations, prePro, min_metacells, reduction_key, dim_str_list, \
        n_features, n_waypoint_eigs, min_iter, max_iter, k_knn = args
    try:
        logger.info(f"Identify metacells using SEACells. gamma = {gamma}")
        for anno in adata.obs[annotations].unique():
            adata_label = adata[adata.obs[annotations] == anno,]
            n_SEACells = round(len(adata_label)/gamma)
                
            if n_SEACells < min_metacells:
                n_SEACells = min_metacells
                    
            if n_SEACells == 1:
                adata_label.obs['SEACell'] = anno + "-" + "SEACell-1"
                if anno == adata.obs[annotations].unique()[0]:
                    seacells_res = adata_label.obs["SEACell"]
                else:
                    seacells_res = pd.concat([seacells_res,adata_label.obs["SEACell"]])
                continue
                
                #print("Identify "+ str(n_SEACells) + " metacells using SEACells...")
            logger.info(f"Identify {n_SEACells} metacells using SEACells...")

            if (reduction_key == 'X_svd'):
                prePro = False
                if (reduction_key not in adata_label.obsm.keys()):
                    raise Exception("No SVD reduction for scATAC data!")
            elif (prePro or reduction_key not in adata_label.obsm.keys()):
                    #print("Preprocess the data...")
                    #print("Normalize cells and compute highly variable genes...")
                logger.info("Preprocess the data...")
                logger.info("Normalize cells and compute highly variable genes...")
                    
                sc.pp.normalize_per_cell(adata_label)
                sc.pp.log1p(adata_label)
                sc.pp.highly_variable_genes(adata_label, n_top_genes=n_features)
                
                    #print("Compute principal components")
                logger.info("Compute principal components")

                sc.tl.pca(adata_label, n_comps=int(dim_str_list[1]), use_highly_variable=True)
                reduction_key = "X_pca"
                
            build_kernel_on = reduction_key
            logger.info("done build_kernel_on")
            dim_lwb = int(dim_str_list[0])-1  # The first dimension of SVD for scATAC should be removed beforehand
            logger.info("done dim_lwb")
            dim_upb = min(int(dim_str_list[1]), adata_label.obsm[build_kernel_on].shape[1])
            logger.info("done dim_upb")
            adata_label.obsm[build_kernel_on] = adata_label.obsm[build_kernel_on][:,range(dim_lwb, dim_upb)]
            logger.info("done adata_label.obsm[build_kernel_on]")
            
            min_metacells = min(min_metacells,adata_label.n_obs)
            
            
            if n_SEACells < n_waypoint_eigs:
                n_waypoint_eigs_label = n_SEACells
            else:
                n_waypoint_eigs_label = n_waypoint_eigs
            
                #n_waypoint = min(n_SEACells, 5)

            model = SEACells.core.SEACells(adata_label, 
            build_kernel_on=build_kernel_on, 
            n_SEACells=n_SEACells,
            verbose=True,
            n_neighbors = k_knn,
            n_waypoint_eigs=n_waypoint_eigs_label,
            use_sparse=True, #added since its been running >11 hours
            max_franke_wolfe_iters=10, #was 50 before
            convergence_epsilon = 1e-3) #was 1e-5 before
            logger.info("done model")
            
            logger.info("begin constructing kernel matrix")
            model.construct_kernel_matrix()
            model.kernel_matrix = model.kernel_matrix.tocsc()
            logger.info("done construct kernel matrix")
                # M = model.kernel_matrix
            
            logger.info("begin intialize archetypes")
            model.initialize_archetypes()
            logger.info("done initialize archetypes")

            #model.archetypes = np.random.choice(model.n_cells, model.k, replace=False)
            #logger.info("skipped initialize_archetypes; selected random archetypes")
            
            #model.initialize_archetypes()
            #logger.info("done initialize archetypes")
                #SEACells.plot.plot_initialization(ad, model)
            #pr = cProfile.Profile()
            #pr.enable()
            logger.info(f"Entering model.fit (γ={gamma})")
            #model.fit(min_iter=min_iter, max_iter=max_iter)
            model.fit(min_iter=5, max_iter=50)
            logger.info(f"Exited model.fit (γ={gamma})")
            #pr.disable()
            #s = io.StringIO()
            #ps = pstats.Stats(pr, stream=s).sort_stats('cumtime')
            #ps.print_stats(20)                # top 20 functions by cumulative time
            #logger.info("Profiling model.fit:\n%s", s.getvalue())
                #model.fit(min_iter=min_iter, max_iter=max_iter)
                #model.plot_convergence()
            logger.info("done fitting")
            adata_label.obs['SEACell'] = "mc" + str(gamma) +"-"+ anno + "-" + adata_label.obs['SEACell']  
            logger.info("done adding mc_")
            if anno == adata.obs[annotations].unique()[0]:
                seacells_res = adata_label.obs["SEACell"]
            else:
                seacells_res = pd.concat([seacells_res,adata_label.obs["SEACell"]])
            logger.info("done for this anno")
            logger.info(anno)
            

        n_metacells = seacells_res.nunique()
        avg_cells   = len(seacells_res) / n_metacells
        logger.info(f"Done identifing metacells using SEACells. gamma = {gamma}")
        logger.info(f"gamma={gamma}: total metacells={n_metacells}, "
                    f"avg cells per metacell={avg_cells:.2f}")
            
        return gamma, seacells_res

    except Exception as e:
        logger.exception(f"Error processing gamma = {gamma}")
        return gamma, None

#%%
def SEACells_mcRigor(input_file,output_file = 'seacells_cell_membership_rna.csv',
             prePro = False,
             Gamma = range(100, 10, -1),
             reduction_key = "X_pca",
             dim_str = "1:50",
             n_features = 2000,
             annotations = None,
             min_metacells = 1,
             n_waypoint_eigs = 10, # was 10, Number of eigenvalues to consider when initializing metacells
             min_iter=10, #was 10 for covid_b
             max_iter=100, #from 100 for covid_b change to 10 for covid full data 200k around
             k_knn = 30 #was 30
             ):
    logger.info(f'input is "{input_file}"')
    logger.info(f'Gamma include "{[i for i in Gamma]}"')
    logger.info(f'dims are "{dim_str}"')
    logger.info(f'reduction_key is "{reduction_key}"')
    adata = input_file	
	
    adata.X = adata.X.astype("float32")

    ####################
    
    dim_str_list = dim_str.split(":") # range input is interesting when using SEACells for ATAC data for which 1 component is often discarded
    # 1 to given components are used when only one number is given to dim
    if (len(dim_str_list)<2):
        dim_str_list += "1"
        dim_str_list.reverse()
    
    # Copy the counts to ".raw" attribute of the anndata since it is necessary for downstream analysis
    # This step should be performed after filtering
    raw_ad = sc.AnnData(adata.X)
    raw_ad.obs_names, raw_ad.var_names = adata.obs_names, adata.var_names
    adata.raw = raw_ad

    if annotations is None:
        annotations = "SEACell_batch"
        adata.obs["SEACell_batch"] = "allcells"

    cell_membership = pd.DataFrame(index=adata.obs.index)

    ctx = (adata, annotations,prePro, min_metacells, reduction_key,
       dim_str_list, n_features, n_waypoint_eigs,
       min_iter, max_iter, k_knn)

    arg_list = [(g, *ctx) for g in Gamma]
    
    logger.info("about to do pooling")
    with Pool(processes=2) as pool:
        results = pool.map(_run_gamma, arg_list)
    
    logger.info("done with the pooling and got the results")
    
    for gamma, res_series in results:
        if res_series is not None:
            logger.info("this is the gamma of")
            logger.info(gamma)
            cell_membership[str(gamma)] = res_series.reindex(adata.obs_names)
            logger.info("done cell_membership for this gamma")
            logger.info(gamma)

    logger.info(f"Saving membership for {len(cell_membership.columns)} gammas to CSV")    
    #cell_membership.to_csv(output_file)
    
    chunk_size = 10_000
    nrows = len(cell_membership)
    
    for start in range(0, nrows, chunk_size):
        end = min(start + chunk_size, nrows)
        # 'w'rite + header for first chunk, 'a'ppend without header afterwards
        mode   = 'w' if start == 0 else 'a'
        header = (start == 0)

        cell_membership.iloc[start:end].to_csv(
            output_file,
            mode=mode,
            header=header,
            index=True
        )
        logger.info(f"Wrote rows {start}-{end-1} ({end-start} rows)")

    logger.info(f"All {nrows} rows written to {output_file}")
    logger.info(f"CSV save complete — wrote to {output_file}") 
    
    logger.info("All finished!")
    

if __name__ == "__main__":

    start = time.time()
    #input_file = '/scratch/dvl5760/blish_covid.seu.h5ad'
    input_file = '/scratch/dvl5760/GSE156793_scjoint_subsample_prep.h5ad'
    adata = sc.read_h5ad(input_file)
    logger.info(f"done reading adata in {time.time()-start:.1f}s, shape: {adata.shape}")
    

    #adata = adata[:10000, :].copy()


    start = time.time()
    covid_b = adata
#covid_b = adata[(adata.obs['Status'] == 'COVID') & (adata.obs["cell.type.coarse"]=="B")].copy()
#covid_b = adata[adata.obs['Status'] == 'COVID'].copy()
#covid_b = adata[(adata.obs['Status'] != 'COVID') & (adata.obs["cell.type.coarse"]=="B")].copy()
    logger.info(f"selected COVID–B subset in {time.time()-start:.1f}s → {covid_b.shape}")

    #logger.info("begin pre-processing covid_b")
    #sc.pp.filter_cells(covid_b, min_genes=200)
    #sc.pp.filter_genes(covid_b, min_cells=3)
    #logger.info("Shape after filtering: %s", covid_b.shape)


    #X_dense = covid_b.X.A if sparse.issparse(covid_b.X) else covid_b.X
    #num_neg = np.sum(X_dense < 0)
    #logger.warning("Number of negative entries in covid_b.X: %d", num_neg)

# Op#tional fix: clip to zero
    #if num_neg > 0:
    #    logger.warning("Clipping all negative values to 0")
    #    covid_b.X = np.maximum(X_dense, 0)


    #total_counts = covid_b.X.sum(axis=1).A1 if sparse.issparse(covid_b.X) else covid_b.X.sum(axis=1)
    #logger.info("Min/Max total counts before normalization: %.1f / %.1f", total_counts.min(), total_counts.max())
    #logger.info("Number of cells with total count = 0: %d", np.sum(total_counts == 0))


    #sc.pp.normalize_total(covid_b, target_sum=1e4)
    #logger.info("Total-count normalization complete")
    #logger.info("Shape after normalization: %s", covid_b.shape)


#tot#al_counts = covid_b.X.sum(axis=1).A1 if sparse.issparse(covid_b.X) else covid_b.X.sum(axis=1)
#log#ger.info("Min/Max total counts after normalization: %.1f / %.1f", total_counts.min(), total_counts.max())
#log#ger.info("Number of cells with total count = 0: %d", np.sum(total_counts == 0))
#sc.#pp.scale(covid_b, max_value=10)

    #sc.pp.log1p(covid_b)
    #logger.info("Log1p transformation complete")
    #logger.info("Shape after log1p: %s", covid_b.shape)

#sc.#pp.scale(covid_b, max_value=10)
#log#ger.info("Shape after clipping: %s", covid_b.shape)

    #if not sparse.issparse(covid_b.X):
    #    covid_b.X = csr_matrix(covid_b.X)
    #    logger.info("Converted covid_b.X back to sparse matrix")

    #X_dense = covid_b.X.A if sparse.issparse(covid_b.X) else covid_b.X
    #gene_means = np.mean(X_dense, axis=0)
    #safe_genes_mask = ~np.isnan(gene_means) & ~np.isinf(gene_means)

    #if not np.all(safe_genes_mask):
    #    logger.warning("Removing %d genes with NaN or Inf mean before HVG", (~safe_genes_mask).sum())
    #    covid_b = covid_b[:, safe_genes_mask].copy()

    #if covid_b.shape[1] == 0:
    #    logger.error("No genes left after cleaning. Exiting.")
    #    exit(1)

    #sc.pp.highly_variable_genes(covid_b, n_top_genes=2000, flavor='seurat')
    #n_hvg = covid_b.var['highly_variable'].sum()
    #logger.info("Selected %d highly variable genes", n_hvg)
    #covid_b = covid_b[:, covid_b.var['highly_variable']].copy()
    #logger.info("Shape after hvg selection: %s", covid_b.shape)

    #sc.pp.scale(covid_b, max_value=10)
    #logger.info("Scaled gene expression")
    #sc.tl.pca(covid_b, svd_solver='arpack')
    #logger.info("Computed PCA")
    #logger.info("Shape after pca: %s", covid_b.shape)

    #logger.info("done with pre-processing step")

    logger.info(f"covid with b cell: {covid_b.shape}")

#30,32,35,38,42,47,53,60,70,85,106,141,212,425
#20,30,32,35,38,42,47,53,60,70
#100,150,200,250,300,350,400,450,500,550
#550,500,450,400,350,300,250,200,150,100


    logger.info("about to run seacell + mcrigor on covid with b cell")
    start = time.time()
    #SEACells_mcRigor(input_file = covid_b, Gamma = [1000], output_file = '/storage/home/dvl5760/work/SEACells/seacell_default_output/human_fetal_atlas/seacell_default_partition_1000.csv',reduction_key='X_pca')
    #SEACells_mcRigor(input_file = covid_b, Gamma = [550,500], output_file = '/storage/home/dvl5760/work/SEACells/seacell_default_output/human_fetal_atlas/seacell_default_partition_550_500_another.csv',reduction_key='X_pca')
    #SEACells_mcRigor(input_file = covid_b, Gamma = [450,400], output_file = '/storage/home/dvl5760/work/SEACells/seacell_default_output/human_fetal_atlas/seacell_default_partition_450_400.csv',reduction_key='X_pca')
    #SEACells_mcRigor(input_file = covid_b, Gamma = [350,300], output_file = '/storage/home/dvl5760/work/SEACells/seacell_default_output/human_fetal_atlas/seacell_default_partition_350_300.csv',reduction_key='X_pca')
    #SEACells_mcRigor(input_file = covid_b, Gamma = [250,200], output_file = '/storage/home/dvl5760/work/SEACells/seacell_default_output/human_fetal_atlas/seacell_default_partition_250_200.csv',reduction_key='X_pca')
    #SEACells_mcRigor(input_file = covid_b, Gamma = [150,100], output_file = '/storage/home/dvl5760/work/SEACells/seacell_default_output/human_fetal_atlas/seacell_default_partition_150_100.csv',reduction_key='X_pca')
    SEACells_mcRigor(input_file = covid_b, Gamma = [550,500,450,400,350,300,250,200,150,100], output_file = '/storage/home/dvl5760/work/SEACells/seacell_default_output/human_fetal_atlas/seacell_default_partition.csv',reduction_key='X_pca')
    logger.info("done")
    current = time.time()
    logger.info(f"used {current - start} seconds for covid with b cell for mcrigor + seacell")

