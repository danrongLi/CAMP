library(Seurat)
library(foreach)
#registerDoParallel(cores = 4)
library(doMC)
registerDoMC(cores = 4)
#library(doParallel)
#cl <- makeCluster(2)
#registerDoParallel(cl)
library(metacell)


MetaCell1_mcRigor <- function(X,  # gene by cell count matrix
                              cell.metadata = NULL,
                              mc_dir = 'MC1/',
                              mc_id = 'mc1',
                              GENE_USE = NULL,
                              EXCLUDED_GENE_NAMES = NULL,
                              EXCLUDED_GENE_PATTERNS = NULL,
                              CELL_MIN_UMI = 1,
                              k_knn = 50,
                              initial_knn_amp = 6,
                              outlier_detect = T,
                              seed = 123456,
                              filter_gene = T,
                              n_downsamp_gstat = NULL,
                              do_downsamp = T,
                              ...){
  
  N.c <- ncol(X)
  
  if(is.null(colnames(X))){
    warning("colnames(X) is Null, \nGene expression matrix X is expected to have cellIDs as colnames! \nCellIDs will be created automatically in a form 'cell_i' ")
    colnames(X) <- paste("cell", 1:N.c, sep = "_")
  }
  
  cell.ids <- colnames(X)
  
  
  ###
  
  if(!dir.exists(mc_dir)) dir.create(mc_dir, recursive = TRUE)
  scdb_init(mc_dir, force_reinit=T)
  
  scdb_add_mat(id = mc_id, mat = scm_new_matrix(X, cell_metadata = cell.metadata))
  
  if(!dir.exists(paste0(mc_dir, "figs/"))) dir.create(paste0(mc_dir, "figs/"), recursive = TRUE)
  scfigs_init(paste0(mc_dir, "figs/"))
  
  # ignore some bad genes
  mat = scdb_mat(mc_id)
  nms = c(rownames(mat@mat), rownames(mat@ignore_gmat))
  
  if (!is.null(GENE_USE)) {
    bad_genes = setdiff(nms, GENE_USE)
  } else {
    ig_genes = c(grep("^IGJ", nms, v=T), 
                 grep("^IGH",nms,v=T),
                 grep("^IGK", nms, v=T), 
                 grep("^IGL", nms, v=T))
    
    bad_genes = unique(c(grep("^MT-", nms, v=T), grep("^MTMR", nms, v=T), grep("^MTND", nms, v=T),
                         "NEAT1","TMSB4X", "TMSB10", ig_genes))
    
    if (!is.null(EXCLUDED_GENE_NAMES)) bad_genes = unique(c(bad_genes, EXCLUDED_GENE_NAMES))
    if (!is.null(EXCLUDED_GENE_PATTERNS)) bad_genes = unique(c(bad_genes, 
                                                               do.call(c, lapply(EXCLUDED_GENE_PATTERNS, function(gp) grep(gp, nms, v=T)))))
  }
  
  mcell_mat_ignore_genes(new_mat_id=mc_id, mat_id=mc_id, bad_genes, reverse=F) 
  
  mcell_mat_ignore_small_cells(mc_id, mc_id, CELL_MIN_UMI)
  
  if (filter_gene) {
    
    if (!is.null(n_downsamp_gstat)) tgconfig::set_param('scm_n_downsamp_gstat', n_downsamp_gstat, package = 'metacell')
    mcell_add_gene_stat(gstat_id=mc_id, mat_id=mc_id, force=T)
    if (!is.null(n_downsamp_gstat)) tgconfig::set_param('scm_n_downsamp_gstat', NULL, package = 'metacell')
    
    mcell_gset_filter_varmean(gset_id=mc_id, gstat_id=mc_id, T_vm=0.08, force_new=T)
    mcell_gset_filter_cov(gset_id = mc_id, gstat_id=mc_id, T_tot=100, T_top3=2)
    
  } else {
    
    mat = scdb_mat(mc_id)
    gset = rep(1, length(mat@genes))
    names(gset) = mat@genes
    scdb_add_gset(mc_id, gset = gset_new_gset(gset, 'all'))
    
  }
  
  mcell_add_cgraph_from_mat_bknn(mat_id=mc_id, 
                                 gset_id =mc_id, 
                                 graph_id=mc_id,
                                 K=k_knn * initial_knn_amp,
                                 dsamp=do_downsamp)
  
  mcell_coclust_from_graph_resamp(
    coc_id=mc_id, 
    graph_id=mc_id,
    min_mc_size=1, 
    p_resamp=0.75, n_resamp=500)
  
  mcell_mc_from_coclust_balanced(
    coc_id=mc_id, 
    mat_id= mc_id,
    mc_id= mc_id, 
    K=k_knn, min_mc_size=1, alpha=2)
  
  mcell_mc_screen_outliers_1gene_fold(
    new_mc_id = mc_id,
    mc_id     = mc_id,
    mat_id    = mc_id,
    T_lfc     = 3
  )
  mc <- scdb_mc(mc_id)
  
  if (outlier_detect) {
    mc0 <- scdb_mc(mc_id)
    clusts <- unique(mc0@mc[!is.na(mc0@mc)])
    if (length(clusts) > 1) {
      tryCatch({
        mcell_mc_split_filt(
          new_mc_id = mc_id, mc_id = mc_id, mat_id = mc_id,
          T_lfc = 3, plot_mats = FALSE
        )
      }, error = function(e) {
        warning("mcell_mc_split_filt() failed, skipping: ", e$message)
      })
    } else {
      message("Skipping split — only one metacell present.")
    }
  }
  
  mc = scdb_mc(mc_id)
  
  sc_membership = rep(NA, N.c)
  names(sc_membership) = cell.ids
  
  sc_membership[names(mc@mc)] = mc@mc
  
  gamma = round(length(mc@mc) / length(table(sc_membership[!is.na(sc_membership)])))
  
  res = list(gamma = gamma, k_knn = k_knn, sc_membership = sc_membership)
  
  return(res)
}


#seu <- readRDS("/Users/danrongli/Desktop/yo/SEACells/data/blish_covid.seu.rds")
#seu <- readRDS("/scratch/dvl5760/blish_covid.seu.rds")
seu <- readRDS("/scratch/dvl5760/GSE156793_scjoint_subsample_prep.seurat.rds")
seu <- UpdateSeuratObject(seu)

#cells_keep <- WhichCells(seu, expression = Status == "COVID" & cell.type.coarse == "B")
#cells_keep <- WhichCells(seu, expression = Status == "COVID")
#covid_b    <- subset(seu, cells = cells_keep)
covid_b <- seu

DefaultAssay(seu) <- "RNA"
print(Layers(seu[["RNA"]]))                      # what layers are present?

print(dim(GetAssayData(seu, layer="counts")))    # shape of counts
print(dim(GetAssayData(seu, layer="data")))      # shape of data (should be genes x cells)
print(length(Cells(seu)))                        # number of cell IDs

Xc <- GetAssayData(seu, assay = "RNA", layer = "counts")  # sparse dgCMatrix
cat("nonzeros:", length(Xc@x), "\n")
summ <- summary(Xc@x[sample.int(length(Xc@x), min(1e5, length(Xc@x)))])
print(summ)
cat("maxÃ¢â€°Ë†", max(Xc@x), "  meanÃ¢â€°Ë†", mean(Xc@x), "\n")
cat("integer fraction in sampleÃ¢â€°Ë†", mean(abs(Xc@x - round(Xc@x)) < 1e-9), "\n")

DefaultAssay(seu) <- "RNA"

Xnorm <- GetAssayData(seu, layer = "counts")

if (is.null(colnames(Xnorm))) {
  colnames(Xnorm) <- Cells(seu)
}

seu[["RNA"]] <- SetAssayData(seu[["RNA"]], layer = "data", new.data = Xnorm)

print(Layers(seu[["RNA"]]))
print(dim(GetAssayData(seu, layer="data")))   # should be 2000 x 494515
stopifnot(identical(colnames(GetAssayData(seu, layer="data")), Cells(seu)))

X <- GetAssayData(seu, layer = "data")        # genes x cells (sparse)
cell_meta <- seu@meta.data[colnames(X), , drop = FALSE]
stopifnot(identical(rownames(cell_meta), colnames(X)))


#covid_b <- subset(covid_b, subset = nFeature_RNA >= 200)

#keep_genes <- rowSums(covid_b@assays$RNA@counts > 0) >= 3
#covid_b    <- covid_b[keep_genes, ]

#neg_cnt <- sum(covid_b@assays$RNA@counts < 0)
#if (neg_cnt > 0) {
#  warning(paste("Clipping", neg_cnt, "negative values → 0"))
#  m            <- covid_b@assays$RNA@counts
#  m[m < 0]     <- 0
#  covid_b@assays$RNA@counts <- m
#}

#DefaultAssay(covid_b) <- "RNA"
#covid_b <- NormalizeData(covid_b,
#                         assay = "RNA",
#                         normalization.method = "LogNormalize",
#                         scale.factor = 1e4)

#mat        <- as.matrix(GetAssayData(covid_b, assay="RNA", slot="data"))
#gene_means <- rowMeans(mat, na.rm = TRUE)
#keep       <- is.finite(gene_means)
#if (any(!keep)) {
#  warning(sprintf("Dropping %d genes with NaN/Inf", sum(!keep)))
#  covid_b <- covid_b[keep, ]
#}

#covid_b <- FindVariableFeatures(covid_b,
#                                assay = "RNA",
#                                selection.method = "vst",
#                                nfeatures = 2000)
#cat("→ HVGs:", length(VariableFeatures(covid_b)), "\n")

#covid_b <- ScaleData(covid_b,
#                     assay    = "RNA",
#                     features = VariableFeatures(covid_b))

#covid_b <- RunPCA(covid_b,
#                  assay    = "RNA",
#                  features = VariableFeatures(covid_b),
#                  verbose  = FALSE)
#cat("After PCA – Cells:", ncol(covid_b),
#    "Genes:", nrow(covid_b), "\n")

#hvg <- VariableFeatures(covid_b)

#X <- as.matrix(GetAssayData(covid_b,
#                            assay = "RNA",
#                            slot  = "data")[hvg, , drop = FALSE])
#cat(dim(X))
#mode(X) <- "numeric"

#ncells   <- ncol(X)
#cell_meta <- data.frame(
#  batch     = rep(1, ncells),
#  row.names = colnames(X),
#  stringsAsFactors = FALSE
#)

cat("\n=== 🚀 About to launch parallel Metacell loop ===\n")
flush.console()   # force any buffered cat() out to the console

knn_amp_val <- c(1,2,3,4,5)
all_membership <- foreach(amp_val = knn_amp_val,
                          .packages = c("metacell","Seurat"),.verbose=TRUE) %dopar% {
       			start_time <- Sys.time()	
                            cat("[pid", Sys.getpid(), "] starting amp =", amp_val, "\n")
                            #message("[", Sys.getpid(), "] starting amp = ", amp_val)
                            flush.console()
                            
                            tryCatch({
                              res <- MetaCell1_mcRigor(
                                X               = X,
                                cell.metadata   = cell_meta,
                                mc_dir          = file.path("MC1_amp", amp_val),
                                mc_id           = paste0("mc1_amp", amp_val),
                                k_knn           = 50,
                                do_downsamp     = FALSE,
                                n_downsamp_gstat= 1000,
                                filter_gene     = FALSE,
                                outlier_detect  = TRUE,
                                initial_knn_amp = amp_val
                              )
                              as.character(res$sc_membership)
                            }, error = function(e){
                              cat("Error in amp", amp_val, "→", e$message, "\n")
                              NULL
                            })
			    
  if (is.null(res)) {
    NULL
  }
  end_time <- Sys.time()
  elapsed  <- round(difftime(end_time, start_time, units = "secs"), 2)
  cat("[pid", Sys.getpid(), "] amp =", amp_val, "took", elapsed, "seconds\n")

  # now compute and print metacell stats
  membership <- res$sc_membership
  assigned   <- membership[!is.na(membership)]
  n_metacells <- length(unique(assigned))
  avg_cells   <- mean(table(assigned))

  cat("  → Number of metacells:", n_metacells, "\n")
  cat("  → Avg cells per metacell:", round(avg_cells, 2), "\n")
  flush.console()

  # return the raw membership vector
  as.character(membership)
                          }

membership_df <- as.data.frame(all_membership, stringsAsFactors = FALSE)
rownames(membership_df) <- colnames(X)
colnames(membership_df) <- knn_amp_val

#out_csv <- "/storage/home/dvl5760/work/SEACells/customized_metacell/from_local_to_server_methods/covid_healthy/metacell1_membership_amp.csv"
out_csv <- "/storage/home/dvl5760/work/SEACells/customized_metacell/from_local_to_server_methods/human_metacell1/output/metacell1_amp_partitions_new.csv"
write.csv(membership_df, file = out_csv, quote = FALSE, row.names = TRUE)
cat("Wrote metacell1 membership to", out_csv, "\n")

#stopCluster(cl)

