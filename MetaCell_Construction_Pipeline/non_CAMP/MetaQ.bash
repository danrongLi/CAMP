#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=120GB
#SBATCH --partition=open
#SBATCH --time=2-00:00:00

set -euo pipefail

module load anaconda/2023.09
conda activate MetaQ

# 1) Paths & params
RAW_INPUT="/scratch/dvl5760/GSE156793_scjoint_subsample_prep.h5ad"
PROC_OUTPUT="/scratch/dvl5760/GSE156793_scjoint_subsample_prep.h5ad"
SAVE_PREFIX="human_fetal_atlas"
#METAS=(500 1000 1500 2000)
#METAS=(4945 3297 2472 1978 1648 1413 1236 1099 989 899)
METAS=(4945 3297 2472 1978)

## 2) Preprocess in‐line
#echo "[$(date)] ⏳ Preprocessing ${RAW_INPUT}"
#python - <<PYCODE
#import time, logging
#import numpy as np
#import scanpy as sc
#from scipy import sparse
#from scipy.sparse import csr_matrix
#
## setup logger
#logger = logging.getLogger("preproc")
#logger.setLevel(logging.INFO)
#ch = logging.StreamHandler()
#fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
#ch.setFormatter(fmt)
#logger.addHandler(ch)
#
#RAW = "${RAW_INPUT}"
#OUT  = "${PROC_OUTPUT}"
#
#start = time.time()
#adata = sc.read_h5ad(RAW)
#logger.info(f"Read {adata.shape} in {time.time()-start:.1f}s")
#
## your filtering & clipping
#start = time.time()
#covid_b = adata  # or your obs‐based subset
#logger.info(f"Selected subset in {time.time()-start:.1f}s, shape {covid_b.shape}")
#
#logger.info("Begin filtering…")
#sc.pp.filter_cells(covid_b, min_genes=200)
#sc.pp.filter_genes(covid_b, min_cells=3)
#logger.info(f"After QC filter: {covid_b.shape}")
#
## clip negatives
#X = covid_b.X.A if sparse.issparse(covid_b.X) else covid_b.X
#neg = np.sum(X < 0)
#if neg>0:
#    logger.warning(f"Clipping {neg} negative entries")
#    covid_b.X = np.maximum(X, 0)
#
## normalize & log1p
#sc.pp.normalize_total(covid_b, target_sum=1e4)
#sc.pp.log1p(covid_b)
#logger.info("Normalized & log1p")
#
## ensure sparse
#if not sparse.issparse(covid_b.X):
#    covid_b.X = csr_matrix(covid_b.X)
#    logger.info("Converted to sparse")
#
## drop NaN/Inf genes
#means = np.asarray(covid_b.X.mean(axis=0)).ravel()
#mask = (~np.isnan(means)) & (~np.isinf(means))
#if mask.sum() < covid_b.shape[1]:
#    logger.warning(f"Dropping {covid_b.shape[1]-mask.sum()} bad genes")
#    covid_b = covid_b[:, mask].copy()
#
## HVG
#sc.pp.highly_variable_genes(covid_b, n_top_genes=2000, flavor='seurat')
#hvg_count = covid_b.var['highly_variable'].sum()
#logger.info(f"Selected {hvg_count} HVGs")
#covid_b = covid_b[:, covid_b.var.highly_variable].copy()
#logger.info(f"After HVG: {covid_b.shape}")
#
## save
#covid_b.write_h5ad(OUT)
#logger.info(f"Wrote processed data to {OUT}")
#PYCODE
#
#echo "[$(date)] ✅ Preprocessing done"

# 3) Run MetaQ for each metacell size
echo "[$(date)] 🏃‍♂️ Running MetaQ loop"
for M in "${METAS[@]}"; do
  OUT_NAME="${SAVE_PREFIX}_${M}"
  echo "  → metacell_num=${M}  save_name=${OUT_NAME}"
  export CUDA_VISIBLE_DEVICES=""
  python MetaQ.py \
    --data_path "${PROC_OUTPUT}" \
    --data_type RNA \
    --metacell_num "${M}" \
    --save_name "${OUT_NAME}"\
    --device cpu\
    --type_key Main_cluster_name\
    --train_epoch 40 \
    --batch_size 1024 \
    --converge_threshold 2
done

echo "[$(date)] 🎉 All runs complete"
