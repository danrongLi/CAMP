import logging
# optimization
# for parallelizing stuff
from multiprocessing import cpu_count

# --- cap math-library threads to avoid oversubscription (must be before numpy) ---
import os
for v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(v, "1")
# ------------------------------------------------------------------------------


import numpy as np
from joblib import Parallel, delayed
from scipy.sparse import lil_matrix
#from tqdm.notebook import tqdm
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
import scipy.sparse as sp

# get number of cores for multiprocessing
NUM_CORES = cpu_count()


# Create or get the logger
logger = logging.getLogger(__name__)

# Set the level of the logger. This is optional and can be set to other levels (e.g., DEBUG, ERROR)
logger.setLevel(logging.INFO)

# Create a console handler and set the level to INFO
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Create a formatter and set the formatter for the handler
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)

# Add the handler to the logger
logger.addHandler(console_handler)

logger.info('done importing stuff')

##########################################################
# Helper functions for parallelizing kernel construction
##########################################################

#add the same metric function as seen in accessibility file
def outer_product_frobenius(x, y):
    outer = np.outer(x, y)  # x y^T
    identity = np.eye(len(x))  # assuming x and y have same length
    return np.linalg.norm(outer - identity, ord='fro')  # Frobenius norm


def permute_columns_inplace(X):
    X_perm = X.copy()
    for col_index in range(X_perm.shape[1]):
        np.random.shuffle(X_perm[:, col_index])
    return X_perm


def permute_rows_inplace(X):
    X_perm = X.copy()
    for row_index in range(X_perm.shape[0]):
        np.random.shuffle(X_perm[row_index, :])
    return X_perm


def kth_neighbor_distance_fast(dist_csr, k, i):
    """
    Return the k-th value (0-indexed) among the nonzero distances in row i.
    dist_csr: CSR sparse matrix of distances (n x n)
    k: index to select (e.g., median = k//2 in caller)
    i: row index
    """
    start, end = dist_csr.indptr[i], dist_csr.indptr[i+1]
    row_data = dist_csr.data[start:end]   # distances to stored neighbors only
    if row_data.size == 0:
        return 0.0
    # bound in case row has < k+1 items
    kk = k if k < row_data.size else row_data.size - 1
    # O(nnz_row) selection without full sort
    return np.partition(row_data, kk)[kk]


def kth_neighbor_distance(distances, k, i):
    """Returns distance to kth nearest neighbor.

    Distances: sparse CSR matrix
    k: kth nearest neighbor
    i: index of row
    .
    """
    # convert row to 1D array
    row_as_array = distances[i, :].toarray().ravel()

    # number of nonzero elements
    num_nonzero = np.sum(row_as_array > 0)

    # argsort
    kth_neighbor_idx = np.argsort(np.argsort(-row_as_array)) == num_nonzero - k
    return np.linalg.norm(row_as_array[kth_neighbor_idx])


def rbf_for_row(G_csr, X, median_distances, i):
    """
    Compute one sparse RBF row for node i using only its neighbors in G_csr.
    Returns a CSR row of shape (1, n).
    """
    start, end = G_csr.indptr[i], G_csr.indptr[i+1]
    idx = G_csr.indices[start:end]  # neighbor indices of i
    if idx.size == 0:
        return sp.csr_matrix((1, X.shape[0]), dtype=np.float32)

    # squared distances only for neighbors
    xi = X[i]
    diff = X[idx] - xi
    # fast row-wise squared norms
    d2 = np.einsum('ij,ij->i', diff, diff)

    denom = median_distances[i] * median_distances[idx]
    # avoid divide-by-zero
    denom = np.where(denom == 0, 1e-12, denom)

    vals = np.exp(-d2 / denom).astype(np.float32, copy=False)
    # build sparse row directly
    row = sp.csr_matrix((vals, (np.zeros_like(idx), idx)), shape=(1, X.shape[0]), dtype=np.float32)
    return row



def rbf_for_row_previous(G, data, median_distances, i):
    """Helper function for computing radial basis function kernel for each row of the data matrix.

    :param G: (array) KNN graph representing nearest neighbour connections between cells
    :param data: (array) data matrix between which euclidean distances are computed for RBF
    :param median_distances: (array) radius for RBF - the median distance between cell and k nearest-neighbours
    :param i: (int) data row index for which RBF is calculated
    :return: sparse matrix containing computed RBF for row
    """
    # convert row to binary numpy array
    row_as_array = G[i, :].toarray().ravel()

    # compute distances ||x - y||^2 in PC/original X space
    numerator = np.sum(np.square(data[i, :] - data), axis=1, keepdims=False)

    # compute radii - median distance is distance to kth nearest neighbor
    denominator = median_distances[i] * median_distances

    # exp
    full_row = np.exp(-numerator / denominator)

    # masked row - to contain only indices captured by G matrix
    masked_row = np.multiply(full_row, row_as_array)

    return lil_matrix(masked_row)


##########################################################
# Archetypal Analysis Metacell Graph
##########################################################


class SEACellGraph:
    """SEACell graph class."""

    def __init__(self, ad, build_on="X_pca", n_cores: int = -1, verbose: bool = False):
        """SEACell graph class.

        :param ad: (anndata.AnnData) object containing data for which metacells are computed
        :param build_on: (str) key corresponding to matrix in ad.obsm which is used to compute kernel for metacells
                        Typically 'X_pca' for scRNA or 'X_svd' for scATAC
        :param n_cores: (int) number of cores for multiprocessing. If unspecified, computed automatically as
                        number of CPU cores
        :param verbose: (bool) whether or not to suppress verbose program logging
        """
        """Initialize model parameters"""
        # data parameters
        self.n, self.d = ad.obsm[build_on].shape

        # indices of each point
        self.indices = np.array(range(self.n))

        # save data
        self.ad = ad
        self.build_on = build_on

        self.knn_graph = None
        self.sym_graph = None

        # number of cores for parallelization
        if n_cores != -1:
            self.num_cores = n_cores
        else:
            self.num_cores = NUM_CORES

        self.M = None  # similarity matrix
        self.G = None  # graph
        self.T = None  # transition matrix

        # model params
        self.verbose = verbose

    ##############################################################
    # Methods related to kernel + sim matrix construction
    ##############################################################

    def rbf(self, k: int = 15, graph_construction="union"):
        """Initialize adaptive bandwith RBF kernel (as described in C-isomap).

        :param k: (int) number of nearest neighbors for RBF kernel
        :return: (sparse matrix) constructed RBF kernel
        """
        import scanpy as sc
        logger.info("Computing kNN graph using scanpy NN ...")
        if self.verbose:
            print("Computing kNN graph using scanpy NN ...")

        # compute kNN and the distance from each point to its nearest neighbors
        #need to replace the following 2 lines
        sc.pp.neighbors(self.ad, use_rep=self.build_on, n_neighbors=k, knn=True)
        knn_graph_distances = self.ad.obsp["distances"]
        
       # logger.info("adding new code here")
       # #addig the following
       # # Use custom metric and brute-force search
       # if self.build_on is not None and self.build_on in self.ad.obsm:
       #     logger.info(f"Using representation from obsm[{self.build_on}]")
       #     X = self.ad.obsm[self.build_on]
       # else:
       #     logger.info("Using raw X matrix")
       #     X = self.ad.X.toarray() if sp.issparse(self.ad.X) else self.ad.X
       # logger.info("done creating X")
       # 
       # #adding the following line since I need to change the distance metric more, not just using outer_product_forbenius
       # #X_1 = permute_rows_inplace(X)
       # X_perm = permute_columns_inplace(X)
       # #Sigma_tilde = X_perm.T @ X_perm
       # Sigma_tilde = np.corrcoef(X_perm, rowvar=False)
       # #Sigma_tilde = np.cov(X_perm, rowvar=False)
       # 
       # def mimic_mcrigor(x,y):
       #     upper = outer_product_frobenius(x, y)
       #     lower = np.linalg.norm(Sigma_tilde - np.eye(len(x)), ord = 'fro')
       #     return upper/lower
       # 
       # def mimic_mcrigor_2(x,y):
       #     A = np.column_stack((x, y))
       #     upper = np.linalg.norm(A @ A.T - np.eye(len(x)), ord = 'fro')
       #     lower = np.linalg.norm(Sigma_tilde - np.eye(len(x)), ord = 'fro')
       #     return upper/lower
       # 
       # def mimic_mcrigor_3(x,y):
       #     A = np.column_stack((x, y))
       #     upper = np.linalg.norm(A @ A.T - np.eye(len(x)), ord = 'fro')
       #     return upper
       # 
       # def mimic_mcrigor_4(x,y):
       #     A = np.column_stack((x, y))
       #     upper = np.linalg.norm(np.corrcoef(A, rowvar=True) - np.eye(len(x)), ord = 'fro')
       #     lower = np.linalg.norm(Sigma_tilde - np.eye(len(x)), ord = 'fro')
       #     return upper/lower
       # 
       # def mimic_mcrigor_5(x,y):
       #     A = np.column_stack((x, y))
       #     upper = np.linalg.norm(np.cov(A, rowvar=True) - np.eye(len(x)), ord = 'fro')
       #     lower = np.linalg.norm(Sigma_tilde - np.eye(len(x)), ord = 'fro')
       #     return upper/lower
       # 
       # 
       # def mimic_mcrigor_6(x,y):
       #     A = np.column_stack((x, y))
       #     upper = np.linalg.norm(np.cov(A, rowvar=True) - np.eye(len(x)), ord = 'fro')
       #     #lower = np.linalg.norm(Sigma_tilde - np.eye(len(x)), ord = 'fro')
       #     return upper
       # 
       # def mimic_mcrigor_7(x,y):
       #     upper = np.linalg.norm(np.correlate(x,y) - np.eye(len(x)), ord = 'fro')
       #     lower = np.linalg.norm(Sigma_tilde - np.eye(len(x)), ord = 'fro')
       #     return upper/lower        
       # 
       # 
       # 
       # 
       # try:
       #     logger.info("starting NearestNeighbors fitting")
       #     #nbrs = NearestNeighbors(n_neighbors=k, metric=outer_product_frobenius, algorithm='brute')
       #     nbrs = NearestNeighbors(n_neighbors=k, metric=mimic_mcrigor, algorithm='brute')
       #     #nbrs = NearestNeighbors(n_neighbors=k, metric=mimic_mcrigor_2, algorithm='brute')
       #     #nbrs = NearestNeighbors(n_neighbors=k, metric=mimic_mcrigor_4, algorithm='brute')
       #     #nbrs = NearestNeighbors(n_neighbors=k, metric=mimic_mcrigor_5, algorithm='brute')
       #     #nbrs = NearestNeighbors(n_neighbors=k, metric=mimic_mcrigor_3, algorithm='brute')
       #     #nbrs = NearestNeighbors(n_neighbors=k, metric=mimic_mcrigor_6, algorithm='brute')
       #     nbrs.fit(X)
       #     logger.info("done with fitting")
       # except Exception as e:
       #     logger.error(f"Error during NearestNeighbors fit: {e}")
       # 
       # 
       # distances, indices = nbrs.kneighbors(X)
       # logger.info("getting distances, indices")
       # 
       # n_cells = X.shape[0]
       # rows = np.repeat(np.arange(n_cells), k)
       # cols = indices.flatten()
       # dists = distances.flatten()
       # logger.info("done flattening")
       # 
       # # Create sparse matrix of distances
       # knn_graph_distances = sp.csr_matrix((dists, (rows, cols)), shape=(n_cells, n_cells))
       # #done adding for now
       # logger.info("done adding here")
        
     #   logger.info("About to binarize knn_graph")
     #   # Binarize distances to get connectivity
     #   knn_graph = knn_graph_distances.copy()
     #   #going to replace the next 1 line with another line
     #   knn_graph[knn_graph != 0] = 1
     #   #knn_graph.data = np.ones_like(knn_graph.data)
     #   #logger.info("done running the replaced line")
     #   # Include self as neighbour
     #   knn_graph.setdiag(1)
      
        #logger.info("About to binarize knn_graph")
        knn_graph = knn_graph_distances.tocsr(copy=True)  # ensure CSR for fast ops
        #logger.info("done knn_graph_distances.tocsr(copy=True)")
        knn_graph.data[:] = 1                              # O(nnz) binarization (fast)
        #logger.info("done knn_graph.data[:] = 1")
        knn_graph.eliminate_zeros()                        # keep structure clean
        #logger.info("done knn_graph.eliminate_zeros()")
        knn_graph.setdiag(1)                               # include self as neighbor
        #logger.info("done knn_graph.setdiag(1)")
        #logger.info(f"nnz(distances)={knn_graph_distances.nnz}, nnz(knn)={knn_graph.nnz}")

       # logger.info("adding 2 lines more")
       # #adding 2 lines more
       # self.ad.obsp["distances"] = knn_graph_distances
       # self.ad.obsp["connectivities"] = knn_graph
       # #done adding
       # logger.info("done adding here")

        self.knn_graph = knn_graph
        logger.info("Computing radius for adaptive bandwidth kernel...")
        #logger.info("done self.knn_graph = knn_graph")
        if self.verbose:
            print("Computing radius for adaptive bandwidth kernel...")

            # compute median distance for each point amongst k-nearest neighbors
        with Parallel(n_jobs=self.num_cores, backend="threading") as parallel:
            median = k // 2
            median_distances = parallel(
                delayed(kth_neighbor_distance_fast)(knn_graph_distances, median, i)
                for i in tqdm(range(self.n))
            )
        #logger.info("done parallel things")

        # convert to numpy array
        median_distances = np.array(median_distances)
        #logger.info("done median_distances")
        
        logger.info("Making graph symmetric...")
        if self.verbose:
            print("Making graph symmetric...")

        print(
            f"Parameter graph_construction = {graph_construction} being used to build KNN graph..."
        )
        if graph_construction == "union":
            #sym_graph = (knn_graph + knn_graph.T > 0).astype(float)
            sym_graph = knn_graph.maximum(knn_graph.T)            # avoids >0 compare & cast
        elif graph_construction in ["intersect", "intersection"]:
            #knn_graph = (knn_graph > 0).astype(float)
            sym_graph = knn_graph.multiply(knn_graph.T)
        else:
            raise ValueError(
                f"Parameter graph_construction = {graph_construction} is not valid. \
             Please select `union` or `intersection`"
            )

        self.sym_graph = sym_graph
        #logger.info("done self.sym_graph")
        logger.info("Computing RBF kernel...")
        if self.verbose:
            print("Computing RBF kernel...")
        

        sym_graph = self.sym_graph.tocsr(copy=False)
        X = np.asarray(self.ad.obsm[self.build_on], dtype=np.float32, order='C')

        with Parallel(n_jobs=self.num_cores, backend="threading") as parallel:
            similarity_matrix_rows = parallel(
                delayed(rbf_for_row)(sym_graph, X, median_distances, i)
                for i in tqdm(range(self.n))
            )

        #with Parallel(n_jobs=self.num_cores, backend="threading") as parallel:
        #    similarity_matrix_rows = parallel(
        #        delayed(rbf_for_row)(
        #            sym_graph, self.ad.obsm[self.build_on], median_distances, i
        #        )
        #        for i in tqdm(range(self.n))
        #    )
        #logger.info("done second parallel things")
        
        logger.info("Building similarity LIL matrix...")

        if self.verbose:
            print("Building similarity LIL matrix...")

        #similarity_matrix = lil_matrix((self.n, self.n))
        #logger.info("done lil_matrix((self.n, self.n))")
        #for i in tqdm(range(self.n)):
        #    similarity_matrix[i] = similarity_matrix_rows[i]
        #logger.info("done for i in tqdm(range(self.n))")

        #if self.verbose:
        #    print("Constructing CSR matrix...")

        #self.M = (similarity_matrix).tocsr()
        
        # stack rows in C instead of Python-loop assignment
        rows_csr = [row if sp.isspmatrix_csr(row) else row.tocsr() for row in similarity_matrix_rows]
        similarity_matrix = sp.vstack(rows_csr, format="csr")
        similarity_matrix = similarity_matrix.astype(np.float32, copy=False)
        
        logger.info("Constructing CSR matrix...")

        if self.verbose:
            print("Constructing CSR matrix...")
        
        self.M = similarity_matrix
        
        
        return self.M
