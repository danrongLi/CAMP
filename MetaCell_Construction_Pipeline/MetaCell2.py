import logging
#%%
##%%time

import os
# turn off HDF5 file locking on NFS
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import numpy as np
import pandas as pd
import scanpy as sc
import metacells as mc

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
def setup_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(fmt)
        logger.addHandler(ch)
    return logger

logger = setup_logger()
logger.info('done importing stuff')

# %%
def prepare_run_metacell0_9(adata, 
                    proj_name = "metacells",
                    pre_filter_cells = True,
                    EXCLUDED_GENE_NAMES = ["XIST", "MALAT1", "NEAT1"], 
                    EXCLUDED_GENE_PATTERNS = ["MT-.*"],
                    ADDITIONAL_CELLS_MASKS = None,   #name of a logical column in .obs with True for cells to discard
                    PROPERLY_SAMPLED_MIN_CELL_TOTAL = 100,
                    PROPERLY_SAMPLED_MAX_CELL_TOTAL = 20000,
                    PROPERLY_SAMPLED_MAX_EXCLUDED_GENES_FRACTION = 0.25,
                    LATERAL_GENE_NAMES = [
    "ACSM3", "ANP32B", "APOE", "AURKA", "B2M", "BIRC5", "BTG2", "CALM1", "CD63", "CD69", "CDK4",
    "CENPF", "CENPU", "CENPW", "CH17-373J23.1", "CKS1B", "CKS2", "COX4I1", "CXCR4", "DNAJB1",
    "DONSON", "DUSP1", "DUT", "EEF1A1", "EEF1B2", "EIF3E", "EMP3", "FKBP4", "FOS", "FOSB", "FTH1",
    "G0S2", "GGH", "GLTSCR2", "GMNN", "GNB2L1", "GPR183", "H2AFZ", "H3F3B", "HBM", "HIST1H1C",
    "HIST1H2AC", "HIST1H2BG", "HIST1H4C", "HLA-A", "HLA-B", "HLA-C", "HLA-DMA", "HLA-DMB",
    "HLA-DPA1", "HLA-DPB1", "HLA-DQA1", "HLA-DQB1", "HLA-DRA", "HLA-DRB1", "HLA-E", "HLA-F", "HMGA1",
    "HMGB1", "HMGB2", "HMGB3", "HMGN2", "HNRNPAB", "HSP90AA1", "HSP90AB1", "HSPA1A", "HSPA1B",
    "HSPA6", "HSPD1", "HSPE1", "HSPH1", "ID2", "IER2", "IGHA1", "IGHA2", "IGHD", "IGHG1", "IGHG2",
    "IGHG3", "IGHG4", "IGHM", "IGKC", "IGKV1-12", "IGKV1-39", "IGKV1-5", "IGKV3-15", "IGKV4-1",
    "IGLC2", "IGLC3", "IGLC6", "IGLC7", "IGLL1", "IGLL5", "IGLV2-34", "JUN", "JUNB", "KIAA0101",
    "LEPROTL1", "LGALS1", "LINC01206", "LTB", "MCM3", "MCM4", "MCM7", "MKI67", "MT2A", "MYL12A",
    "MYL6", "NASP", "NFKBIA", "NUSAP1", "PA2G4", "PCNA", "PDLIM1", "PLK3", "PPP1R15A", "PTMA",
    "PTTG1", "RAN", "RANBP1", "RGCC", "RGS1", "RGS2", "RGS3", "RP11-1143G9.4", "RP11-160E2.6",
    "RP11-53B5.1", "RP11-620J15.3", "RP5-1025A1.3", "RP5-1171I10.5", "RPS10", "RPS10-NUDT3", "RPS11",
    "RPS12", "RPS13", "RPS14", "RPS15", "RPS15A", "RPS16", "RPS17", "RPS18", "RPS19", "RPS19BP1",
    "RPS2", "RPS20", "RPS21", "RPS23", "RPS24", "RPS25", "RPS26", "RPS27", "RPS27A", "RPS27L",
    "RPS28", "RPS29", "RPS3", "RPS3A", "RPS4X", "RPS4Y1", "RPS4Y2", "RPS5", "RPS6", "RPS6KA1",
    "RPS6KA2", "RPS6KA2-AS1", "RPS6KA3", "RPS6KA4", "RPS6KA5", "RPS6KA6", "RPS6KB1", "RPS6KB2",
    "RPS6KC1", "RPS6KL1", "RPS7", "RPS8", "RPS9", "RPSA", "RRM2", "SMC4", "SRGN", "SRSF7", "STMN1",
    "TK1", "TMSB4X", "TOP2A", "TPX2", "TSC22D3", "TUBA1A", "TUBA1B", "TUBB", "TUBB4B", "TXN", "TYMS",
    "UBA52", "UBC", "UBE2C", "UHRF1", "YBX1", "YPEL5", "ZFP36", "ZWINT"
],
                    LATERAL_GENE_PATTERNS = ["RP[LS].*"],
                    NOISY_GENE_NAMES = [
    "CCL3", "CCL4", "CCL5", "CXCL8", "DUSP1", "FOS", "G0S2", "HBB", "HIST1H4C", "IER2", "IGKC",
    "IGLC2", "JUN", "JUNB", "KLRB1", "MT2A", "RPS26", "RPS4Y1", "TRBC1", "TUBA1B", "TUBB"
],
                    seed = 123456
):
                  
  mc.ut.set_name(adata, proj_name)
  adata.X.sort_indices()
  
  mc.pl.exclude_genes(
    adata,
    excluded_gene_names=EXCLUDED_GENE_NAMES, 
    excluded_gene_patterns=EXCLUDED_GENE_PATTERNS,
    random_seed=seed
  )
  
  if pre_filter_cells:
      logger.info("Filter cells using standard MetaCell pipeline...")
      # compute number of UMIs in excluded genes 
      mc.tl.compute_excluded_gene_umis(adata)
      
      if ADDITIONAL_CELLS_MASKS is not None:
          ADDITIONAL_CELLS_MASKS = adata.obs[ADDITIONAL_CELLS_MASKS]
  
      # exclude cells based on totals UMIs and number of UMIs in excluded genes + additional cell annotation provided by the user (additional_cells_masks)
      mc.pl.exclude_cells(
      adata,
      properly_sampled_min_cell_total=PROPERLY_SAMPLED_MIN_CELL_TOTAL,
      properly_sampled_max_cell_total=PROPERLY_SAMPLED_MAX_CELL_TOTAL,
      properly_sampled_max_excluded_genes_fraction=PROPERLY_SAMPLED_MAX_EXCLUDED_GENES_FRACTION,
      additional_cells_masks=ADDITIONAL_CELLS_MASKS
      )
  else:
      mc.pl.exclude_cells(
      adata,
      properly_sampled_min_cell_total=None,
      properly_sampled_max_cell_total=None,
      properly_sampled_max_excluded_genes_fraction=None,
      additional_cells_masks=None
      )
      
      
  
  clean = mc.pl.extract_clean_data(adata, name="hca_bm.one-pass.clean")
  mc.ut.top_level(clean) 
  logger.info(f"Clean: {clean.n_obs} cells, {clean.n_vars} genes")
  
  # Define lateral and noisy genes 
  mc.pl.mark_lateral_genes(
      clean,
      lateral_gene_names=LATERAL_GENE_NAMES,
      lateral_gene_patterns=LATERAL_GENE_PATTERNS,
  )
  mc.pl.mark_noisy_genes(clean, noisy_gene_names=NOISY_GENE_NAMES)
  
  return(clean)



# %%
def MetaCell2_mcRigor(input_file = None,
                      output_file = 'mc2_cell_membership_rna.csv',
              pre_filter_cells = False,
              Gamma = range(100, 10, -1),
              annotations = None,
              min_metacell_size = 1,
              yml_config = None,
              seed = 123456
              ):

    #logger.info('input is "', input_file)
    #logger.info('Gamma include "', [i for i in Gamma])
    #logger.info('Pre filter cells is', pre_filter_cells)
    
    if isinstance(input_file, str) and input_file.endswith(".h5ad"):
        adata = sc.read_h5ad(input_file)
        
        if adata.raw is not None:
            adata.X = adata.raw  # we only load raw counts, We always normalize .X prior to compute PCA if prePro is asked or reduction_key absent  
            del adata.raw

    #elif isinstance(input_file, str):
    #    adata = sc.read_csv(input_file + 'counts.csv')
    #    metadata = pd.read_csv(input_file + 'metadata.csv', index_col = 0)
    #    adata.obs = metadata

    #    from scipy import sparse
    #    sparse_X = sparse.csr_matrix(adata.X)
    #    adata.X = sparse_X

    else:
        adata = input_file
        logger.info("Shape: %s", adata.shape)
    # The dtype of X is no longer set to float32 in scanpy. 
    # While anndata2ri produces float64, the majority of h5ad objects available online are float32.
    # We choose to set the type to float32
    
    adata.X = adata.X.astype("float32")
    
    if yml_config is not None:
        logger.info("Using MetaCell2 with parameters from"+ yml_config)
        with open(yml_config) as file:
            mc2_params = yaml.safe_load(file)
            
        if mc2_params["ADDITIONAL_CELLS_MASKS"] == "None":
            mc2_params["ADDITIONAL_CELLS_MASKS"] = None
        
        clean = prepare_run_metacell0_9(
          adata,
          pre_filter_cells = pre_filter_cells,
          EXCLUDED_GENE_NAMES = mc2_params["EXCLUDED_GENE_NAMES"], 
          EXCLUDED_GENE_PATTERNS = mc2_params["EXCLUDED_GENE_PATTERNS"],
          ADDITIONAL_CELLS_MASKS = mc2_params["ADDITIONAL_CELLS_MASKS"],
          PROPERLY_SAMPLED_MIN_CELL_TOTAL =  mc2_params["PROPERLY_SAMPLED_MIN_CELL_TOTAL"],
          PROPERLY_SAMPLED_MAX_CELL_TOTAL = mc2_params["PROPERLY_SAMPLED_MAX_CELL_TOTAL"],
          PROPERLY_SAMPLED_MAX_EXCLUDED_GENES_FRACTION = mc2_params["PROPERLY_SAMPLED_MAX_EXCLUDED_GENES_FRACTION"],
          LATERAL_GENE_NAMES = mc2_params["LATERAL_GENE_NAMES"],
          LATERAL_GENE_PATTERNS = mc2_params["LATERAL_GENE_PATTERNS"],
          NOISY_GENE_NAMES = mc2_params["NOISY_GENE_NAMES"],
          seed = mc2_params["seed"])
          
    else:
        clean = prepare_run_metacell0_9(
          adata,
          pre_filter_cells = pre_filter_cells)
    
    logger.info("done with cleaning, then lets add pca")
    sc.pp.scale(adata, max_value=10)
    logger.info("Scaled gene expression")
    sc.tl.pca(adata, svd_solver='arpack')
    logger.info("Computed PCA")
    logger.info("Shape after pca: %s", adata.shape)

    cell_membership = pd.DataFrame(index=clean.obs.index)
    
    mc.pl.set_max_parallel_piles(1)

    for gamma in Gamma:
        try:
            logger.info("Identify metacells using MetaCell2. gamma = " + str(gamma))  
            # Adapt number of parallel piles to the available memory 
            #max_parallel_piles = mc.pl.guess_max_parallel_piles(clean)
            #mc.pl.set_max_parallel_piles(max_parallel_piles)
            mc.pl.divide_and_conquer_pipeline(
            clean,
            target_metacell_size = gamma,
            min_metacell_size = min_metacell_size,
            random_seed = seed)

            #Store membership
            clean.obs['membership'] = [str(i+1) if i >= 0 else np.nan for i in clean.obs["metacell"]]
            clean.obs['membership'] = 'mc' + str(gamma) + '-' + clean.obs['membership']
            cell_membership[str(gamma)] = clean.obs['membership']
            
            
            membership_counts = clean.obs['membership'].value_counts()
            num_metacells    = membership_counts.shape[0]
            avg_cells        = membership_counts.mean()
            logger.info(
                f"Gamma {gamma}: number of metacells = {num_metacells}, "
                f"avg cells/metacell = {avg_cells:.2f}"
            )
            
            
            cell_membership.to_csv(output_file)

            # Aggregate metacells 
            # adata_mc = mc.pl.collect_metacells(clean, name='metacells', random_seed = seed)
        except:
            logger.info('Something went wrong for gamma = ' + str(gamma))
    
    logger.info("All finished!")

    


# # %%
# input_file = '/home/pan/wrapup/cd34/input/cd34_multiome_rna.h5ad'
# MetaCell2_mcRigor(input_file=input_file, 
#                   output_file = 'mc2_cell_membership_rna.csv',
#                   Gamma=range(100,10,-1))

# %%
#gamma_list = [20,30,32,35,38,42,47,53,60,70]
gamma_list = [2,4,6,8,10,12,14,16,18]
#input_file = '/scratch/dvl5760/blish_covid.seu.h5ad'
input_file = '/scratch/dvl5760/GSE156793_scjoint_subsample_prep.h5ad'
start = time.time()
adata = sc.read_h5ad(input_file)
logger.info(f"done reading adata in {time.time()-start:.1f}s, shape: {adata.shape}")

start = time.time()
#covid_b = adata[(adata.obs['Status'] == 'COVID') & (adata.obs["cell.type.coarse"]=="B")].copy()
#covid_b = adata[adata.obs['Status'] == 'COVID'].copy()
#covid_b = adata[(adata.obs['Status'] != 'COVID') & (adata.obs["cell.type.coarse"]=="B")].copy()
covid_b = adata
logger.info(f"selected COVID subset in {time.time()-start:.1f}s with shape {covid_b.shape}")

#logger.info("begin pre-processing covid_b")
#sc.pp.filter_cells(covid_b, min_genes=200)
#sc.pp.filter_genes(covid_b, min_cells=3)
#logger.info("Shape after filtering: %s", covid_b.shape)

#X_dense = covid_b.X.A if sparse.issparse(covid_b.X) else covid_b.X
#num_neg = np.sum(X_dense < 0)
#logger.warning("Number of negative entries in covid_b.X: %d", num_neg)
#if num_neg > 0:
#    logger.warning("Clipping all negative values to 0")
#    covid_b.X = np.maximum(X_dense, 0)

#total_counts = covid_b.X.sum(axis=1).A1 if sparse.issparse(covid_b.X) else covid_b.X.sum(axis=1)
#logger.info("Min/Max total counts before normalization: %.1f / %.1f", total_counts.min(), total_counts.max())
#logger.info("Number of cells with total count = 0: %d", np.sum(total_counts == 0))

#sc.pp.normalize_total(covid_b, target_sum=1e4)
#logger.info("Total-count normalization complete")
#logger.info("Shape after normalization: %s", covid_b.shape)

#sc.pp.log1p(covid_b)
#logger.info("Log1p transformation complete")
#logger.info("Shape after log1p: %s", covid_b.shape)

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





MetaCell2_mcRigor(input_file=covid_b, Gamma=gamma_list,
                  output_file='/storage/home/dvl5760/work/SEACells/customized_metacell/human_fetal_atlas/output/metacell2_membership_small_gamma.csv')

# %%
