#library(Seurat)
#library(doParallel)
#library(SuperCell)
# if(packageVersion("Seurat") >= 5) {options(Seurat.object.assay.version = "v3"); message("you are using seurat v5 with assay option v3")}
library(Seurat)
library(SuperCell)
library(Matrix)
library(dplyr)
library(doMC)
registerDoMC(cores = 4)
library(foreach)



SuperCell_mcRigor <- function(X,
                              reduction_key = c("X_pca", "X_svd"),
                              genes.use = NULL,
                              genes.exclude = NULL,
                              cell.annotation = NULL,
                              cell.split.condition = NULL,
                              n.var.genes = min(1000, nrow(X)),
                              gamma = 10,
                              k.knn = 5,
                              do.scale = TRUE,
                              n.pc = 50,
                              fast.pca = TRUE,
                              do.approx = TRUE,
                              approx.N = 20000,
                              block.size = 10000,
                              seed = 12345,
                              igraph.clustering = c("walktrap", "louvain"),
                              return.singlecell.NW = TRUE,
                              return.hierarchical.structure = TRUE,
                              ...){
  
  N.c <- ncol(X)
  reduction_key <- match.arg(reduction_key)
  
  if(gamma > 100 & N.c < 100000){
    warning(paste0("Graining level (gamma = ", gamma, ") seems to be very large! Please, consider using smaller gamma, the suggested range is 10-50."))
  }
  
  if(is.null(rownames(X))){
    if(!(is.null(genes.use) | is.null(genes.exclude))){
      stop("rownames(X) is Null \nGene expression matrix X is expected to have genes as rownames")
    } else {
      warning("colnames(X) is Null, \nGene expression matrix X is expected to have genes as rownames! \ngenes will be created automatically in a form 'gene_i' ")
      rownames(X) <- paste("gene", 1:nrow(X), sep = "_")
    }
  }
  
  if(is.null(colnames(X))){
    warning("colnames(X) is Null, \nGene expression matrix X is expected to have cellIDs as colnames! \nCellIDs will be created automatically in a form 'cell_i' ")
    colnames(X) <- paste("cell", 1:N.c, sep = "_")
  }
  
  cell.ids <- colnames(X)
  
  keep.genes    <- setdiff(rownames(X), genes.exclude)
  X             <- X[keep.genes,]
  
  
  if(do.approx & approx.N >= N.c){
    do.approx <- FALSE
    warning("approx.N is larger or equal to the number of single cells, thus, an exact simplification will be performed")
  }
  
  if(do.approx & (approx.N < round(N.c/gamma))){
    approx.N <- round(N.c/gamma)
    warning(paste("approx.N is set to N.SC", approx.N))
  }
  
  if(do.approx & ((N.c/gamma) > (approx.N/3))){
    warning("approx.N is not much larger than desired number of super-cells, so an approximate simplification may take londer than an exact one!")
  }
  
  if(do.approx){
    set.seed(seed)
    approx.N            <- min(approx.N, N.c)
    presample           <- sample(1:N.c, size = approx.N, replace = FALSE)
    presampled.cell.ids <- cell.ids[sort(presample)]
    rest.cell.ids       <- setdiff(cell.ids, presampled.cell.ids)
  } else {
    presampled.cell.ids <- cell.ids
    rest.cell.ids       <- c()
  }
  
  
  if (reduction_key == 'X_svd'){
    
    temp_assay <- Signac::CreateChromatinAssay(counts = X[, presampled.cell.ids],
                                               sep = c(':', '-'))
    
    temp_sobj <- Seurat::CreateSeuratObject(counts = temp_assay,
                                            assay = 'peaks')
    temp_sobj <- Signac::RunTFIDF(object = temp_sobj)
    temp_sobj <- Signac::FindTopFeatures(object = temp_sobj, min.cutoff = 'q0')
    temp_sobj <- Signac::RunSVD(object = temp_sobj, n = n.pc + 1)
    
    X.for.graph <- temp_sobj@reductions$lsi@cell.embeddings[, -c(1)]
    
  } else {
    
    if(is.null(genes.use)){
      n.var.genes <- min(n.var.genes, nrow(X))
      if(N.c > 50000){
        set.seed(seed)
        idx         <- sample(N.c, 50000)
        gene.var    <- apply(X[,idx], 1, stats::var)
      } else {
        gene.var    <- apply(X, 1, stats::var)
      }
      
      genes.use   <- names(sort(gene.var, decreasing = TRUE))[1:n.var.genes]
    }
    
    if(length(intersect(genes.use, genes.exclude)) > 0){
      stop("Sets of genes.use and genes.exclude have non-empty intersection")
    }
    
    genes.use <- genes.use[genes.use %in% rownames(X)]
    X <- X[genes.use,]
    
    X.for.pca            <- Matrix::t(X[genes.use, presampled.cell.ids])
    if(do.scale){ X.for.pca            <- scale(X.for.pca) }
    X.for.pca[is.na(X.for.pca)] <- 0
    
    if(is.null(n.pc[1]) | min(n.pc) < 1){stop("Please, provide a range or a number of components to use: n.pc")}
    if(length(n.pc)==1) n.pc <- 1:n.pc
    
    if(fast.pca & (N.c < 1000)){
      warning("Normal pca is computed because number of cell is low for irlba::irlba()")
      fast.pca <- FALSE
    }
    
    if(!fast.pca){
      PCA.presampled          <- stats::prcomp(X.for.pca, rank. = max(n.pc), scale. = F, center = F)
    } else {
      set.seed(seed)
      PCA.presampled          <- irlba::irlba(X.for.pca, nv = max(n.pc, 25))
      PCA.presampled$x        <- PCA.presampled$u %*% diag(PCA.presampled$d)
      PCA.presampled$rotation <- PCA.presampled$v
    }
    
    X.for.graph <- PCA.presampled$x[,n.pc]
  }
  
  
  sc.nw <- SuperCell::build_knn_graph(
    X = X.for.graph,
    k = k.knn, from = "coordinates",
    #use.nn2 = use.nn2,
    dist_method = "euclidean",
    #directed = directed,
    #DoSNN = DoSNN,
    #pruning = pruning,
    #which.snn = which.snn,
    #kmin = kmin,
    ...
  )
  
  
  #simplify
  
  k   <- round(N.c/gamma)
  
  if(igraph.clustering[1] == "walktrap"){
    g.s              <- igraph::cluster_walktrap(sc.nw$graph.knn)
    g.s$membership   <- igraph::cut_at(g.s, k)
    
  } else if(igraph.clustering[1] == "louvain") {
    warning(paste("igraph.clustering =", igraph.clustering, ", gamma is ignored"))
    g.s    <- igraph::cluster_louvain(sc.nw$graph.knn)
    
  } else {
    stop(paste("Unknown clustering method (", igraph.clustering, "), please use louvain or walkrtap"))
  }
  
  membership.presampled        <- g.s$membership
  names(membership.presampled) <- presampled.cell.ids
  
  ## Split super-cells containing cells from different annotations or conditions
  if(!is.null(cell.annotation) | !is.null(cell.split.condition)){
    if(is.null(cell.annotation)) cell.annotation <- rep("a", N.c)
    if(is.null(cell.split.condition)) cell.split.condition <- rep("s", N.c)
    names(cell.annotation) <- names(cell.split.condition) <- cell.ids
    
    split.cells <- interaction(cell.annotation[presampled.cell.ids], cell.split.condition[presampled.cell.ids], drop = TRUE)
    
    membership.presampled.intr <- interaction(membership.presampled, split.cells, drop = TRUE)
    membership.presampled <- as.numeric(membership.presampled.intr)
    names(membership.presampled) <- presampled.cell.ids
  }
  
  
  
  SC.NW                        <- igraph::contract(sc.nw$graph.knn, membership.presampled)
  if(!do.approx){
    SC.NW                        <- igraph::simplify(SC.NW, remove.loops = T, edge.attr.comb="sum")
  }
  
  
  if(do.approx){
    
    PCA.averaged.SC      <- as.matrix(Matrix::t(supercell_GE(t(PCA.presampled$x[,n.pc]), groups = membership.presampled)))
    X.for.roration       <- Matrix::t(X[genes.use, rest.cell.ids])
    
    
    
    if(do.scale){ X.for.roration <- scale(X.for.roration) }
    X.for.roration[is.na(X.for.roration)] <- 0
    
    
    membership.omitted   <- c()
    if(is.null(block.size) | is.na(block.size)) block.size <- 10000
    
    N.blocks <- length(rest.cell.ids)%/%block.size
    if(length(rest.cell.ids)%%block.size > 0) N.blocks <- N.blocks+1
    
    
    if(N.blocks>0){
      for(i in 1:N.blocks){ # compute knn by blocks
        idx.begin <- (i-1)*block.size + 1
        idx.end   <- min(i*block.size,  length(rest.cell.ids))
        
        cur.rest.cell.ids    <- rest.cell.ids[idx.begin:idx.end]
        
        PCA.ommited          <- X.for.roration[cur.rest.cell.ids,] %*% PCA.presampled$rotation[, n.pc] ###
        
        D.omitted.subsampled <- proxy::dist(PCA.ommited, PCA.averaged.SC) ###
        
        membership.omitted.cur        <- apply(D.omitted.subsampled, 1, which.min) ###
        names(membership.omitted.cur) <- cur.rest.cell.ids ###
        
        membership.omitted   <- c(membership.omitted, membership.omitted.cur)
      }
    }
    
    membership.all_       <- c(membership.presampled, membership.omitted)
    membership.all        <- membership.all_
    
    
    names_membership.all <- names(membership.all_)
    ## again split super-cells containing cells from different annotation or split conditions
    if(!is.null(cell.annotation) | !is.null(cell.split.condition)){
      
      split.cells <- interaction(cell.annotation[names_membership.all],
                                 cell.split.condition[names_membership.all], drop = TRUE)
      
      
      membership.all.intr <- interaction(membership.all_, split.cells, drop = TRUE)
      
      membership.all      <- as.numeric(membership.all.intr)
      
    }
    
    
    SC.NW                        <- igraph::simplify(SC.NW, remove.loops = T, edge.attr.comb="sum")
    names(membership.all) <- names_membership.all
    membership.all <- membership.all[cell.ids]
    
  } else {
    membership.all       <- membership.presampled[cell.ids]
  }
  membership       <- membership.all
  
  supercell_size   <- as.vector(table(membership))
  
  igraph::E(SC.NW)$width         <- sqrt(igraph::E(SC.NW)$weight/10)
  
  if(igraph::vcount(SC.NW) == length(supercell_size)){
    igraph::V(SC.NW)$size          <- supercell_size
    igraph::V(SC.NW)$sizesqrt      <- sqrt(igraph::V(SC.NW)$size)
  } else {
    igraph::V(SC.NW)$size          <- as.vector(table(membership.all_))
    igraph::V(SC.NW)$sizesqrt      <- sqrt(igraph::V(SC.NW)$size)
    warning("Supercell graph was not splitted")
  }
  
  
  
  res <- list(graph.supercells = SC.NW,
              gamma = gamma,
              N.SC = length(unique(membership)),
              membership = membership,
              supercell_size = supercell_size,
              genes.use = genes.use,
              simplification.algo = igraph.clustering[1],
              do.approx = do.approx,
              n.pc = n.pc,
              k.knn = k.knn,
              sc.cell.annotation. = cell.annotation,
              sc.cell.split.condition. = cell.split.condition
  )
  
  if(return.singlecell.NW){res$graph.singlecell <- sc.nw$graph.knn}
  if(!is.null(cell.annotation) | !is.null(cell.split.condition)){
    res$SC.cell.annotation. <- supercell_assign(cell.annotation, res$membership)
    res$SC.cell.split.condition. <- supercell_assign(cell.split.condition, res$membership)
  }
  
  if(igraph.clustering[1] == "walktrap" & return.hierarchical.structure)  res$h_membership <- g.s
  
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

DefaultAssay(covid_b) <- "RNA"

cat("Assays:\n"); print(Assays(covid_b))
if ("layers" %in% slotNames(covid_b[["RNA"]]) || inherits(covid_b[["RNA"]]@layers, "Layers")) {
  cat("Layers present in RNA:\n"); print(Layers(covid_b[["RNA"]]))
}
cat("Cells:", length(Cells(covid_b)), "\n")


get_layer_safe <- function(obj, layer_name) {
  out <- tryCatch(GetAssayData(obj, assay = "RNA", layer = layer_name),
                  error = function(e) NULL)
  if (is.null(out)) {
    slot_name <- switch(layer_name, counts = "counts", data = "data", scale = "scale.data", layer_name)
    out <- tryCatch(GetAssayData(obj, assay = "RNA", slot = slot_name),
                    error = function(e) NULL)
  }
  out
}


X_counts <- get_layer_safe(covid_b, "counts")
X_data   <- get_layer_safe(covid_b, "data")


cat("counts dim: "); if (!is.null(X_counts)) print(dim(X_counts)) else cat("NULL\n")
cat("data   dim: "); if (!is.null(X_data))   print(dim(X_data))   else cat("NULL\n")

if (is.null(X_counts) || any(dim(X_counts) == 0)) stop("counts matrix is missing/empty.")
if (!is.null(X_data) && all(dim(X_data) > 0)) {
  message("Note: 'data' layer exists, but per your setup we will use 'counts' as the source matrix.")
}

X <- X_counts  # genes x cells (dgCMatrix)

if (is.null(colnames(X))) colnames(X) <- Cells(covid_b)
if (is.null(rownames(X))) rownames(X) <- rownames(get_layer_safe(covid_b, "counts"))

X <- X[, Cells(covid_b), drop = FALSE]
stopifnot(identical(colnames(X), Cells(covid_b)))

if (length(X@x)) {
  s <- X@x[sample.int(length(X@x), min(1e5, length(X@x)))]
  int_frac <- mean(abs(s - round(s)) < 1e-9)  # << 1 if log/float
  rng <- range(s, finite = TRUE)
  cat(sprintf("Integer fractionâ‰ˆ %.4f; value rangeâ‰ˆ [%.3f, %.3f]\n", int_frac, rng[1], rng[2]))
  if (!is.na(int_frac) && int_frac > 0.95 && rng[2] > 50) {
    warning("Matrix looks like raw counts (mostly integers, large max). If thatâ€™s not intended, fix upstream.")
  }
}

if (length(VariableFeatures(covid_b)) == 0) {
  cat("Finding variable features (vst, 2000)...\n")
  covid_b <- FindVariableFeatures(covid_b, selection.method = "vst", nfeatures = 2000, verbose = FALSE)
}
hvgs <- intersect(VariableFeatures(covid_b), rownames(X))
stopifnot(length(hvgs) > 0)
cat("Usable HVGs:", length(hvgs), "\n")

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

#X <- as.matrix(GetAssayData(covid_b,
#                            assay = "RNA",
#                            slot  = "data"))
#mode(X) <- "numeric"


#gammas <- c(70, 60, 53, 47, 42, 38, 35, 32, 30, 20)

gammas <- c(550,500,450,400,350,300,250,200,150,100)
#gammas <- c(20)

all_membership <- list()
all_counts     <- integer(length(gammas))
names(all_counts) <- as.character(gammas)


parallel_res <- foreach(g = gammas,.packages = c("Seurat","SuperCell","Matrix","dplyr")) %dopar% {

#for (g in gammas) {
 
  cat("Running gamma =", g, "\n")
  t0 <- Sys.time()
  res <- SuperCell_mcRigor(X, gamma = g)
  t1 <- Sys.time()
  cat("  → Time:", round(difftime(t1, t0, units = "secs"), 2), "seconds\n")
  
  # Extract membership vector
  membership <- res$membership
  mc_count   <- length(unique(membership))
  cat(sprintf("  → γ (cells per supercell) = %d; total super-cells = %d\n", 
              g, mc_count))
  #all_membership[[ as.character(g) ]] <- as.character(membership)
  #all_counts[ as.character(g) ]         <- mc_count
 	
  list(gamma     = g,
        membership = membership,
        mc_count   = mc_count)

  # Ensure it's named by cell IDs and stored as character
  #all_membership[[as.character(g)]] <- as.character(membership)
}

for (res in parallel_res) {
	g_char <- as.character(res$gamma)
	all_membership[[g_char]] <- res$membership
	all_counts[g_char]       <- res$mc_count
}

membership_df <- as.data.frame(all_membership)
rownames(membership_df) <- colnames(X)
colnames(membership_df) <- as.character(all_counts)

#out_csv <- "/storage/home/dvl5760/work/SEACells/customized_metacell/from_local_to_server_methods/covid_healthy/supercell_membership.csv"
out_csv <- "/storage/home/dvl5760/work/SEACells/customized_metacell/from_local_to_server_methods/human_supercell/supercell_membership_approx_new.csv"
write.csv(membership_df, file = out_csv, quote = FALSE, row.names = TRUE)
cat("Wrote supercell membership to", out_csv, "\n")

