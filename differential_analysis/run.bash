#!/usr/bin/env bash
#SBATCH --job-name=camp_pbmc_matched_v3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180GB
#SBATCH --account=ykk5167_cr_default
#SBATCH --partition=basic
#SBATCH --time=48:00:00
#SBATCH --output=camp_pbmc_matched_v3_%j.out
#SBATCH --error=camp_pbmc_matched_v3_%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-${SCRIPT_DIR}}}"
PYTHON_SCRIPT="${PROJECT_ROOT}/de.py"
NATIVE_DE_PLOT_SCRIPT="${PROJECT_ROOT}/de_plot.py"
SEACELLS_SOURCE="${SEACELLS_SOURCE:-${PROJECT_ROOT}/SEACells}"
cd "${PROJECT_ROOT}"

# Override any of these values when submitting, for example:
# INPUT_H5AD=/new/path/data.h5ad OUTPUT_DIR=/new/path/results sbatch run.bash
INPUT_H5AD="${INPUT_H5AD:-/storage/home/dvl5760/scratch/blish_covid.seu.h5ad}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/results_pbmc_native_fig4_fig5_v2}"
CONDA_ENV="${CONDA_ENV:-current_env}"
# A task-specific name prevents an unrelated exported STAGE variable from
# silently changing which experiments run. Default: the count-controlled
# Fig. 4 and Fig. 5 analyses plus their MetaQ-style panels.
# To redraw only the MetaQ-style panels from completed checkpoints, submit:
#   PBMC_RUN_STAGE=metaq_plot sbatch run.bash
# To generate only the new donor-blocked B-lineage outputs from the existing
# Fig. 4 checkpoints (no metacell or mapping recomputation), submit:
#   PBMC_RUN_STAGE=blineage sbatch run.bash
# To redraw only the focused paper panels after the B-lineage stage is complete:
#   PBMC_RUN_STAGE=manuscript_plot sbatch run.bash
# This writes manuscript_additional_experiments_matched_v3: the main-text
# panels compare the prespecified default CAMP1 with all competitors, while
# CAMP1-4 and every requested diagnostic remain in the supplement.
# To run the corrected equal-compression comparison:
#   PBMC_RUN_STAGE=matched sbatch run.bash
# To reproduce the native-resolution differential-expression section and all
# of its individual manuscript/supplementary figures from raw inputs:
#   PBMC_RUN_STAGE=native_de sbatch run.bash
PBMC_RUN_STAGE="${PBMC_RUN_STAGE:-matched}"
DEVICE="${DEVICE:-cpu}"
AUTO_INSTALL_DEPS="${AUTO_INSTALL_DEPS:-1}"
INSTALL_ONLY="${INSTALL_ONLY:-0}"

CELLTYPE_KEY="${CELLTYPE_KEY:-auto}"
BATCH_KEY="${BATCH_KEY:-auto}"
DONOR_KEY="${DONOR_KEY:-auto}"
CONDITION_KEY="${CONDITION_KEY:-auto}"
REFERENCE_CONDITION="${REFERENCE_CONDITION:-auto}"

MAPPING_EPOCHS="${MAPPING_EPOCHS:-1000}"
RANDOM_SEED="${RANDOM_SEED:-0}"
DPI="${DPI:-600}"
METAQ_MARKER_GENE="${METAQ_MARKER_GENE:-MS4A1}"
METAQ_MARKER_CELLTYPE="${METAQ_MARKER_CELLTYPE:-B}"
METAQ_FOCUS_CELLTYPE="${METAQ_FOCUS_CELLTYPE:-CD8eff T}"
METAQ_TOP_CELLTYPES="${METAQ_TOP_CELLTYPES:-6}"
METAQ_TOP_GENES_PER_CELLTYPE="${METAQ_TOP_GENES_PER_CELLTYPE:-10}"
if [[ "${PBMC_RUN_STAGE}" == "native_de" ]]; then
  DEFAULT_FIG4_TARGET_METACELLS="1000"
else
  DEFAULT_FIG4_TARGET_METACELLS="750 1000 1250"
fi
FIG4_TARGET_METACELLS="${FIG4_TARGET_METACELLS:-${DEFAULT_FIG4_TARGET_METACELLS}}"
FIG4_PRIMARY_TARGET="${FIG4_PRIMARY_TARGET:-1000}"
read -r -a FIG4_TARGETS <<< "${FIG4_TARGET_METACELLS}"
DE_REDUCTION_RATES="${DE_REDUCTION_RATES:-8 10 12}"
DE_PRIMARY_REDUCTION_RATE="${DE_PRIMARY_REDUCTION_RATE:-10}"
read -r -a DE_RATES <<< "${DE_REDUCTION_RATES}"
BLINEAGE_OUTPUT_SUBDIR="${BLINEAGE_OUTPUT_SUBDIR:-b_lineage_heterogeneity_matched_counts_v2}"
BLINEAGE_CONDITION="${BLINEAGE_CONDITION:-COVID}"
BLINEAGE_CELLTYPES_CSV="${BLINEAGE_CELLTYPES_CSV:-B,Class-switched B,IgA PB,IgG PB}"
BLINEAGE_DONORS_CSV="${BLINEAGE_DONORS_CSV:-auto}"
BLINEAGE_MIN_CELLS_PER_STATE="${BLINEAGE_MIN_CELLS_PER_STATE:-20}"
BLINEAGE_NEIGHBORS="${BLINEAGE_NEIGHBORS:-15}"
BLINEAGE_MARKER_GENES="${BLINEAGE_MARKER_GENES:-MS4A1 IGHD TCL1A MZB1 XBP1 PRDM1 CD38 IGHA1 IGHG1}"
BLINEAGE_PLOT_GENES="${BLINEAGE_PLOT_GENES:-MS4A1 IGHD MZB1 IGHA1 IGHG1}"
IFS=',' read -r -a BLINEAGE_CELLTYPES <<< "${BLINEAGE_CELLTYPES_CSV}"
IFS=',' read -r -a BLINEAGE_DONORS <<< "${BLINEAGE_DONORS_CSV}"
read -r -a BLINEAGE_MARKERS <<< "${BLINEAGE_MARKER_GENES}"
read -r -a BLINEAGE_PLOT_MARKERS <<< "${BLINEAGE_PLOT_GENES}"
REPLACE_FIG4_RESULTS="${REPLACE_FIG4_RESULTS:-0}"
REPLACE_DE_RESULTS="${REPLACE_DE_RESULTS:-0}"

case "${PBMC_RUN_STAGE}" in
  all|compute|fig4|de|native_de|matched|plot|metaq_plot|blineage|manuscript_plot|inspect) ;;
  *)
    echo "Invalid PBMC_RUN_STAGE='${PBMC_RUN_STAGE}'" >&2
    exit 2
    ;;
esac

if [[ "${OUTPUT_DIR}" == "${PROJECT_ROOT}/results_pbmc_fig4_fig5" ]]; then
  echo "Refusing the legacy mixed-results folder: ${OUTPUT_DIR}" >&2
  echo "Use the default results_pbmc_native_fig4_fig5_v2 folder." >&2
  exit 2
fi

if command -v module >/dev/null 2>&1; then
  module load anaconda/2023.09
fi

if command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base)"
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}"
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
fi

# Prevent a previously exported site-packages path from mixing two conda
# environments with the same name. The project-local SEACells path is added
# explicitly later, after dependency verification.
unset PYTHONPATH
unset PYTHONHOME
hash -r
echo "Activated conda prefix: ${CONDA_PREFIX:-unknown}"
python -c "import os, sys; assert os.path.realpath(sys.prefix) == os.path.realpath(os.environ['CONDA_PREFIX']), (sys.prefix, os.environ['CONDA_PREFIX']); print('Using Python executable:', sys.executable)"
python -m pip --version

# Install only missing optional packages into current_env. Existing working
# Scanpy/NumPy/Pandas/SciPy packages are deliberately left unchanged.
if [[ "${AUTO_INSTALL_DEPS}" == "1" ]]; then
  if ! python -c "import pkg_resources" >/dev/null 2>&1; then
    echo "Installing setuptools==80.9.0 for legacy pkg_resources compatibility"
    python -m pip install --no-input --upgrade "setuptools==80.9.0"
  fi

  if ! python -c "from importlib.metadata import version; import harmonypy; assert version('harmonypy') == '0.0.6'" >/dev/null 2>&1; then
    echo "Installing paper-matched harmonypy==0.0.6 into ${CONDA_ENV}"
    python -m pip install --no-input --upgrade "harmonypy==0.0.6"
  fi

  if ! python -c "import torch; from torch import nn" >/dev/null 2>&1; then
    echo "Installing CPU PyTorch 2.1.1 into ${CONDA_ENV}"
    python -m pip install --no-input "torch==2.1.1" \
      --index-url https://download.pytorch.org/whl/cpu
  fi

  if ! python -c "import umap" >/dev/null 2>&1; then
    echo "Installing umap-learn==0.5.5 into ${CONDA_ENV}"
    python -m pip install --no-input "umap-learn==0.5.5"
  fi

  if ! python -c "import igraph, louvain, leidenalg" >/dev/null 2>&1; then
    echo "Installing compatible graph-clustering packages into ${CONDA_ENV}"
    python -m pip install --no-input \
      "igraph==0.11.8" \
      "louvain==0.8.2" \
      "leidenalg==0.10.2"
  fi

  if ! python -c "import joblib, tqdm" >/dev/null 2>&1; then
    echo "Installing SEACells helper dependencies into ${CONDA_ENV}"
    python -m pip install --no-input "joblib>=1.2" "tqdm>=4.64"
  fi
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_DYNAMIC=FALSE
export MPLCONFIGDIR="${OUTPUT_DIR}/.matplotlib"
export NUMBA_CACHE_DIR="${OUTPUT_DIR}/.numba_cache"
mkdir -p "${MPLCONFIGDIR}" "${NUMBA_CACHE_DIR}"
echo "PBMC primary protocol: exact matched compression v3"
echo "Run stage: ${PBMC_RUN_STAGE}"
echo "Exact Fig. 4 metacell grid: ${FIG4_TARGETS[*]}"
echo "Primary MetaQ-style marker resolution: ${FIG4_PRIMARY_TARGET} metacells"
echo "Exact Fig. 5 within-stratum compression grid: ${DE_RATES[*]} fold"
echo "Primary MetaQ-style DE compression: ${DE_PRIMARY_REDUCTION_RATE} fold"
echo "B-lineage output subfolder: ${BLINEAGE_OUTPUT_SUBDIR}"
echo "Results directory: ${OUTPUT_DIR}"
if [[ "${PBMC_RUN_STAGE}" == "native_de" ]]; then
  echo "Native DE protocol: original global partitions nearest m=1000, followed only by cell type x donor x condition boundary intersection"
fi

if [[ "${PBMC_RUN_STAGE}" == "plot" || "${PBMC_RUN_STAGE}" == "metaq_plot" || "${PBMC_RUN_STAGE}" == "blineage" || "${PBMC_RUN_STAGE}" == "manuscript_plot" ]]; then
  python "${PYTHON_SCRIPT}" \
    --stage "${PBMC_RUN_STAGE}" \
    --output-dir "${OUTPUT_DIR}" \
    --fig4-variants camp1 camp2 camp3 camp4 \
    --fig4-target-metacells "${FIG4_TARGETS[@]}" \
    --fig4-primary-target "${FIG4_PRIMARY_TARGET}" \
    --de-reduction-rates "${DE_RATES[@]}" \
    --de-primary-reduction-rate "${DE_PRIMARY_REDUCTION_RATE}" \
    --metaq-marker-gene "${METAQ_MARKER_GENE}" \
    --metaq-marker-celltype "${METAQ_MARKER_CELLTYPE}" \
    --metaq-focus-celltype "${METAQ_FOCUS_CELLTYPE}" \
    --metaq-top-celltypes "${METAQ_TOP_CELLTYPES}" \
    --metaq-top-genes-per-celltype "${METAQ_TOP_GENES_PER_CELLTYPE}" \
    --blineage-output-subdir "${BLINEAGE_OUTPUT_SUBDIR}" \
    --blineage-condition "${BLINEAGE_CONDITION}" \
    --blineage-celltypes "${BLINEAGE_CELLTYPES[@]}" \
    --blineage-donors "${BLINEAGE_DONORS[@]}" \
    --blineage-min-cells-per-state "${BLINEAGE_MIN_CELLS_PER_STATE}" \
    --blineage-neighbors "${BLINEAGE_NEIGHBORS}" \
    --blineage-marker-genes "${BLINEAGE_MARKERS[@]}" \
    --blineage-plot-genes "${BLINEAGE_PLOT_MARKERS[@]}" \
    --dpi "${DPI}"
  exit 0
fi

if ! python -c "from importlib.metadata import version; import pkg_resources; import anndata, joblib, leidenalg, louvain, harmonypy, igraph, matplotlib, numpy, pandas, scanpy, scipy, seaborn, sklearn, torch, tqdm, umap; from harmonypy import compute_lisi; assert version('harmonypy') == '0.0.6'; print('Complete dependency check passed | setuptools', version('setuptools'), '| harmonypy', version('harmonypy'), '| scanpy', scanpy.__version__, '| torch', torch.__version__)"; then
  echo "Missing a required Python package in conda environment '${CONDA_ENV}'." >&2
  echo "Automatic installation was ${AUTO_INSTALL_DEPS}. Check the pip output above." >&2
  exit 1
fi

if [[ "${INSTALL_ONLY}" == "1" ]]; then
  echo "Dependency installation and verification completed for ${CONDA_ENV}."
  exit 0
fi

for required_file in \
  "${PYTHON_SCRIPT}" \
  "${INPUT_H5AD}" \
  "${SEACELLS_SOURCE}/SEACells/build_graph.py" \
  "${SEACELLS_SOURCE}/seacell_default_output/covid_healthy/seacell_default_partition.csv" \
  "${SEACELLS_SOURCE}/customized_metacell/from_local_to_server_methods/covid_healthy/metacell1_membership_amp.csv" \
  "${SEACELLS_SOURCE}/customized_metacell/data/covid_healthy/metacell2_membership_small_gamma.csv" \
  "${SEACELLS_SOURCE}/customized_metacell/from_local_to_server_methods/covid_healthy/supercell_membership.csv" \
  "${SEACELLS_SOURCE}/customized_metacell/MetaQ/save/combined_metacell_labels.csv"; do
  if [[ ! -f "${required_file}" ]]; then
    echo "Required input file not found: ${required_file}" >&2
    exit 1
  fi
done

if [[ "${PBMC_RUN_STAGE}" == "native_de" && ! -f "${NATIVE_DE_PLOT_SCRIPT}" ]]; then
  echo "Required native DE plotting script not found: ${NATIVE_DE_PLOT_SCRIPT}" >&2
  exit 1
fi

# Use ${PROJECT_ROOT}/SEACells/SEACells directly, without installing it.
export PYTHONPATH="${SEACELLS_SOURCE}:${PYTHONPATH:-}"
echo "Using bundled SEACells from ${SEACELLS_SOURCE}"
python -c "import importlib.util; p='${SEACELLS_SOURCE}/SEACells/build_graph.py'; s=importlib.util.spec_from_file_location('camp_local_seacells_build_graph', p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); print('SEACells graph import check passed:', p)"

DE_REPLACE_ARGS=()
if [[ "${REPLACE_DE_RESULTS}" == "1" ]]; then
  DE_REPLACE_ARGS+=(--force-de)
fi

FIG4_REPLACE_ARGS=()
if [[ "${REPLACE_FIG4_RESULTS}" == "1" ]]; then
  FIG4_REPLACE_ARGS+=(--force-fig4)
fi

python "${PYTHON_SCRIPT}" \
  --stage "${PBMC_RUN_STAGE}" \
  --input-h5ad "${INPUT_H5AD}" \
  --output-dir "${OUTPUT_DIR}" \
  --celltype-key "${CELLTYPE_KEY}" \
  --batch-key "${BATCH_KEY}" \
  --donor-key "${DONOR_KEY}" \
  --condition-key "${CONDITION_KEY}" \
  --reference-condition "${REFERENCE_CONDITION}" \
  --fig4-variants camp1 camp2 camp3 camp4 \
  --fig4-target-metacells "${FIG4_TARGETS[@]}" \
  --fig4-primary-target "${FIG4_PRIMARY_TARGET}" \
  --de-reduction-rates "${DE_RATES[@]}" \
  --de-primary-reduction-rate "${DE_PRIMARY_REDUCTION_RATE}" \
  --baseline-root "${SEACELLS_SOURCE}" \
  --mapping-epochs "${MAPPING_EPOCHS}" \
  --device "${DEVICE}" \
  --random-seed "${RANDOM_SEED}" \
  --metaq-marker-gene "${METAQ_MARKER_GENE}" \
  --metaq-marker-celltype "${METAQ_MARKER_CELLTYPE}" \
  --metaq-focus-celltype "${METAQ_FOCUS_CELLTYPE}" \
  --metaq-top-celltypes "${METAQ_TOP_CELLTYPES}" \
  --metaq-top-genes-per-celltype "${METAQ_TOP_GENES_PER_CELLTYPE}" \
  --blineage-output-subdir "${BLINEAGE_OUTPUT_SUBDIR}" \
  --blineage-condition "${BLINEAGE_CONDITION}" \
  --blineage-celltypes "${BLINEAGE_CELLTYPES[@]}" \
  --blineage-donors "${BLINEAGE_DONORS[@]}" \
  --blineage-min-cells-per-state "${BLINEAGE_MIN_CELLS_PER_STATE}" \
  --blineage-neighbors "${BLINEAGE_NEIGHBORS}" \
  --blineage-marker-genes "${BLINEAGE_MARKERS[@]}" \
  --blineage-plot-genes "${BLINEAGE_PLOT_MARKERS[@]}" \
  --dpi "${DPI}" \
  "${FIG4_REPLACE_ARGS[@]}" \
  "${DE_REPLACE_ARGS[@]}"

if [[ "${PBMC_RUN_STAGE}" == "native_de" ]]; then
  python "${NATIVE_DE_PLOT_SCRIPT}" \
    --results-root "${OUTPUT_DIR}" \
    --dpi "${DPI}"
fi

echo "PBMC ${PBMC_RUN_STAGE} analysis completed in ${OUTPUT_DIR}."
