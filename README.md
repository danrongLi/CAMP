# 🌟 CAMP: Coreset Accelerated Metacell Partitioning
![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white)
![R](https://img.shields.io/badge/R-4.0%2B-276DC3?logo=r&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Project-Active-brightgreen)


CAMP (**C**oreset **A**ccelerated **M**etacell **P**artitioning) is a scalable, geometry-preserving framework for constructing metacells from large-scale single-cell RNA-seq datasets. CAMP combines coreset theory, archetypal analysis, and streaming similarity computation to produce biologically meaningful metacells in near-linear time.

## 🔍 Overview

CAMP supports four algorithmic variants:

- **CAMP1** — Euclidean coreset with Lloyd-style updates
- **CAMP2** — Linear kernel with vectorized streaming dot-products
- **CAMP3** — Adaptive Gaussian kernel
- **CAMP4** — Mixed Euclidean–kernel similarity model


## 🚀 Features

- Near-linear scalability to >500k cells
- Streaming computation with on-the-fly kernel/distance updates
- Memory-efficient (no full similarity matrices)
- Strong metacell quality metrics
- Full PBMC 44k and Human Fetal Atlas 504k benchmarks included

## 📦 Installation

```bash
git clone https://github.com/danrongLi/CAMP.git
cd CAMP
```
## ▶️ Usage

CAMP is implemented as standalone Python scripts.
To run any CAMP variant, simply open the repository in an IDE such as PyCharm, CLion, or VSCode, and run the desired script directly.
```bash
python camp1.py
python camp2.py
python camp3.py
python camp4.py
```

## 📂 Dataset Access

[Here](https://drive.google.com/drive/folders/1N0xDb8TC0ZPDTmAk6TE612inGBsPtofD?usp=sharing) we provide the pre-processed data that are ready to use.


## 📂 Folder Explanation

[MetaCell_Construction_Pipeline](MetaCell_Construction_Pipeline): Contains the pipelines for running CAMP variants, as well as state-of-the-art methods, including SEACells, MetaQ, SuperCell, MetaCell and MetaCell2.

[Calculate_Quality_Metrics](Calculate_Quality_Metrics): Contains the code for calculating the metacell quality metrics for all methods.

[Our_Optimization_SEACells](Our_Optimization_SEACells): Contains the 2 files we optimized for SEACells method.

[rare_cell_recovery](rare_cell_recovery): Contains rare cell recovery experiments.

[robustness](robustness): Contains robustness experiments.


