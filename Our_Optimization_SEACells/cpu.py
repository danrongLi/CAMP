import logging
import copy

import numpy as np
import palantir
import pandas as pd
import scipy.sparse as sp
from scipy.sparse import csr_matrix, save_npz
from scipy.sparse.linalg import norm
from sklearn.preprocessing import normalize
from tqdm import tqdm

try:
    from . import build_graph
except ImportError:
    import build_graph


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
#logger.addHandler(console_handler)

if not logger.handlers:
    logger.addHandler(console_handler)

logger.info('inside the cpu.py file')


class SEACellsCPU:
    """CPU Implementation of SEACells algorithm.

    This implementation uses fast kernel archetypal analysis to find SEACells - groupings
    of cells that represent highly granular, distinct cell states. SEACells are found by solving a convex optimization
    problem that minimizes the residual sum of squares between the kernel matrix and the weighted sum of the archetypes.

    Modifies annotated data matrix in place to include SEACell assignments in ad.obs['SEACell']
    """

    def __init__(
        self,
        ad,
        build_kernel_on: str,
        n_SEACells: int,
        verbose: bool = True,
        n_waypoint_eigs: int = 10,
        n_neighbors: int = 15,
        convergence_epsilon: float = 1e-3,
        l2_penalty: float = 0,
        max_franke_wolfe_iters: int = 50,
    ):
        """CPU Implementation of SEACells algorithm.

        :param ad: (AnnData) annotated data matrix
        :param build_kernel_on: (str) key corresponding to matrix in ad.obsm which is used to compute kernel for metacells
                                Typically 'X_pca' for scRNA or 'X_svd' for scATAC
        :param n_SEACells: (int) number of SEACells to compute
        :param verbose: (bool) whether to suppress verbose program logging
        :param n_waypoint_eigs: (int) number of eigenvectors to use for waypoint initialization
        :param n_neighbors: (int) number of nearest neighbors to use for graph construction
        :param convergence_epsilon: (float) convergence threshold for Franke-Wolfe algorithm
        :param l2_penalty: (float) L2 penalty for Franke-Wolfe algorithm
        :param max_franke_wolfe_iters: (int) maximum number of iterations for Franke-Wolfe algorithm

        Class Attributes:
            ad: (AnnData) annotated data matrix
            build_kernel_on: (str) key corresponding to matrix in ad.obsm which is used to compute kernel for metacells
            n_cells: (int) number of cells in ad
            k: (int) number of SEACells to compute
            n_waypoint_eigs: (int) number of eigenvectors to use for waypoint initialization
            waypoint_proportion: (float) proportion of cells to use for waypoint initialization
            n_neighbors: (int) number of nearest neighbors to use for graph construction
            max_FW_iter: (int) maximum number of iterations for Franke-Wolfe algorithm
            verbose: (bool) whether to suppress verbose program logging
            l2_penalty: (float) L2 penalty for Franke-Wolfe algorithm
            RSS_iters: (list) list of residual sum of squares at each iteration of Franke-Wolfe algorithm
            convergence_epsilon: (float) algorithm converges when RSS < convergence_epsilon * RSS(0)
            convergence_threshold: (float) convergence threshold for Franke-Wolfe algorithm
            kernel_matrix: (csr_matrix) kernel matrix of shape (n_cells, n_cells)
            K: (csr_matrix) dot product of kernel matrix with itself, K = K @ K.T
            archetypes: (list) list of cell indices corresponding to archetypes
            A_: (csr_matrix) matrix of shape (k, n) containing final assignments of cells to SEACells
            B_: (csr_matrix) matrix of shape (n, k) containing archetype weights
            A0: (csr_matrix) matrix of shape (k, n) containing initial assignments of cells to SEACells
            B0: (csr_matrix) matrix of shape (n, k) containing initial archetype weights
        """
        print("Welcome to SEACells!")
        self.ad = ad
        self.build_kernel_on = build_kernel_on
        self.n_cells = ad.shape[0]

        if not isinstance(n_SEACells, int):
            try:
                n_SEACells = int(n_SEACells)
            except ValueError:
                raise ValueError(
                    f"The number of SEACells specified must be an integer type, not {type(n_SEACells)}"
                )

        self.k = n_SEACells

        self.n_waypoint_eigs = n_waypoint_eigs
        self.waypoint_proportion = 1
        self.n_neighbors = n_neighbors

        self.max_FW_iter = max_franke_wolfe_iters
        self.verbose = verbose
        self.l2_penalty = l2_penalty

        self.RSS_iters = []
        self.convergence_epsilon = convergence_epsilon
        self.convergence_threshold = None

        # Parameters to be initialized later in the model
        self.kernel_matrix = None
        self.K = None

        # Archetypes as list of cell indices
        self.archetypes = None

        self.A_ = None
        self.B_ = None
        self.A0 = None
        self.B0 = None

        return

    def add_precomputed_kernel_matrix(self, K):
        """Add precomputed kernel matrix to SEACells object.

        :param K: (np.ndarray) kernel matrix of shape (n_cells, n_cells)
        :return: None.
        """
        assert K.shape == (
            self.n_cells,
            self.n_cells,
        ), f"Dimension of kernel matrix must be n_cells = ({self.n_cells},{self.n_cells}), not {K.shape} "
        self.kernel_matrix = K

        # Pre-compute dot product
        self.K = self.kernel_matrix @ self.kernel_matrix.T

    def construct_kernel_matrix(
        self, n_neighbors: int = None, graph_construction="union"
    ):
        """Construct kernel matrix from data matrix using PCA/SVD and nearest neighbors.

        :param n_neighbors: (int) number of nearest neighbors to use for graph construction.
                            If none, use self.n_neighbors, which has a default value of 15.
        :param graph_construction: (str) method for graph construction. Options are 'union' or 'intersection'.
                                    Default is 'union', where the neighborhood graph is made symmetric by adding an edge
                                    (u,v) if either (u,v) or (v,u) is in the neighborhood graph. If 'intersection', the
                                    neighborhood graph is made symmetric by adding an edge (u,v) if both (u,v) and (v,u)
                                    are in the neighborhood graph.
        :return: None.
        """
        # input to graph construction is PCA/SVD
        kernel_model = build_graph.SEACellGraph(
            self.ad, self.build_kernel_on, verbose=self.verbose
        )

        # K is a sparse matrix representing input to SEACell alg
        if n_neighbors is None:
            n_neighbors = self.n_neighbors

        M = kernel_model.rbf(n_neighbors, graph_construction=graph_construction)
        self.kernel_matrix = M

        # Pre-compute dot product
        self.K = self.kernel_matrix @ self.kernel_matrix.T

        return

    def initialize_archetypes(self):
        """Initialize B matrix which defines cells as SEACells.

        Uses waypoint analysis for initialization into to fully
        cover the phenotype space, and then greedily selects the remaining cells (if redundant cells are selected by
        waypoint analysis).

        Modifies self.archetypes in-place with the indices of cells that are used as initialization for archetypes.

        By default, the proportion of cells selected by waypoint analysis is 1. This can be changed by setting the
        waypoint_proportion parameter in the SEACells object. For example, setting waypoint_proportion = 0.5 will
        select half of the cells by waypoint analysis and half by greedy selection.
        """
        k = self.k

        if self.waypoint_proportion > 0:
            waypoint_ix = self._get_waypoint_centers(k)
            waypoint_ix = np.random.choice(
                waypoint_ix,
                int(len(waypoint_ix) * self.waypoint_proportion),
                replace=False,
            )
            from_greedy = self.k - len(waypoint_ix)
            if self.verbose:
                print(
                    f"Selecting {len(waypoint_ix)} cells from waypoint initialization."
                )
        else:
            from_greedy = self.k

        greedy_ix = self._get_greedy_centers(n_SEACells=from_greedy + 10)
        if self.verbose:
            print(f"Selecting {from_greedy} cells from greedy initialization.")

        if self.waypoint_proportion > 0:
            all_ix = np.hstack([waypoint_ix, greedy_ix])
        else:
            all_ix = np.hstack([greedy_ix])

        unique_ix, ind = np.unique(all_ix, return_index=True)
        all_ix = unique_ix[np.argsort(ind)][:k]
        self.archetypes = all_ix

    def initialize(self, initial_archetypes=None, initial_assignments=None):
        """Initialize the model by initializing the B matrix.

        The method constructs archetypes from a convex combination of cells) and
        the A matrix (defines assignments of cells to archetypes.

        Assumes the kernel matrix has already been constructed. B matrix is of shape (n_cells, n_SEACells) and A matrix
        is of shape (n_SEACells, n_cells).

        :param initial_archetypes: (np.ndarray) initial archetypes to use for initialization. If None, use waypoint
                                     analysis and greedy selection to initialize archetypes.
        :param initial_assignments: (np.ndarray) initial assignments to use for initialization. If None, use
                                        random initialization.
        :return: None
        """
        logger.info("ok im inside this initialize function")
        if self.K is None:
            raise RuntimeError(
                "Must first construct kernel matrix before initializing SEACells."
            )
        K = self.K
        # initialize B (update this to allow initialization from RRQR)
        n = K.shape[0]

        if initial_archetypes is not None:
            if self.verbose:
                print("Using provided list of initial archetypes")
            self.archetypes = initial_archetypes

        if self.archetypes is None:
            self.initialize_archetypes()
        self.k = len(self.archetypes)
        k = self.k

        # Sparse construction of B matrix
        cols = np.arange(k)
        rows = self.archetypes
        shape = (n, k)
        B0 = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=shape)

        self.B0 = B0
        B = self.B0.copy()
        #logger.info("got B")

        if initial_assignments is not None:
            A0 = initial_assignments
            assert A0.shape == (
                k,
                n,
            ), f"Initial assignment matrix should be of shape (k={k} x n={n})"
            A0 = csr_matrix(A0)
            A0 = normalize(A0, axis=0, norm="l1")

        #else:
        #    logger.info("inside else, begin initialize A0")
        #    # Ultra-sparse A0: assign each cell to s archetypes (s small)
        #    s = 3 if k >= 3 else 1  # 1–3 nonzeros per cell
        #    rows = np.random.randint(0, k, size=(n, s)).reshape(-1)
        #    cols = np.repeat(np.arange(n), s)
        #    vals = np.ones_like(cols, dtype=np.float32)  # or use np.random.rand(cols.size).astype(np.float32)

        #    A0 = csr_matrix((vals, (rows, cols)), shape=(k, n), dtype=np.float32)
        #    A0 = normalize(A0, axis=0, norm="l1")

        #    if self.verbose:
        #        print(f"Sparse A0 initialized with {s} nonzeros per cell.")
        #    logger.info("done initializing A0")

        else:
            # Need to ensure each cell is assigned to at least one archetype
            # Randomly sample roughly 25% of the values between 0 and k
            archetypes_per_cell = int(k * 0.25)
            rows = np.random.randint(0, k, size=(n, archetypes_per_cell)).reshape(-1)
            columns = np.repeat(np.arange(n), archetypes_per_cell)

            A0 = csr_matrix(
                (np.random.random(len(rows)), (rows, columns)), shape=(k, n)
            )
            A0 = normalize(A0, axis=0, norm="l1")

            if self.verbose:
                print("Randomly initialized A matrix.")
            #logger.info("done initializing A0")


        self.A0 = A0
        A = self.A0.copy()
        #logger.info("about to updateA")
        A = self._updateA(B, A)
        #logger.info("done updateA")

        self.A_ = A
        self.B_ = B
        
        #logger.info("about to do the rss things")
        # Create convergence threshold
        RSS = self.compute_RSS(A, B)
        self.RSS_iters.append(RSS)
        #logger.info("done with rss")

        if self.convergence_threshold is None:
            self.convergence_threshold = self.convergence_epsilon * RSS
            if self.verbose:
                print(
                    f"Setting convergence threshold at {self.convergence_threshold:.5f}"
                )
        logger.info("done with this initialize")

    def _get_waypoint_centers(self, n_waypoints=None):
        """Initialize B matrix using waypoint analysis, as described in Palantir.

        From https://www.nature.com/articles/s41587-019-0068-4.

        :param n_waypoints: (int) number of SEACells to initialize using waypoint analysis. If None specified,
                        all SEACells initialized using this method.
        :return: (np.ndarray) indices of cells to use as initial archetypes
        """
        if n_waypoints is None:
            k = self.k
        else:
            k = n_waypoints

        ad = self.ad

        if self.build_kernel_on == "X_pca":
            pca_components = pd.DataFrame(ad.obsm["X_pca"]).set_index(ad.obs_names)
        elif self.build_kernel_on == "X_svd":
            # Compute PCA components from ad object
            pca_components = pd.DataFrame(ad.obsm["X_svd"]).set_index(ad.obs_names)
        else:
            pca_components = pd.DataFrame(ad.obsm[self.build_kernel_on]).set_index(
                ad.obs_names
            )

        print(f"Building kernel on {self.build_kernel_on}")

        if self.verbose:
            print(
                f"Computing diffusion components from {self.build_kernel_on} for waypoint initialization ... "
            )

        dm_res = palantir.utils.run_diffusion_maps(
            pca_components, n_components=self.n_neighbors
        )
        dc_components = palantir.utils.determine_multiscale_space(
            dm_res, n_eigs=self.n_waypoint_eigs
        )
        if self.verbose:
            print("Done.")

        # Initialize SEACells via waypoint sampling
        if self.verbose:
            print("Sampling waypoints ...")
        waypoint_init = palantir.core._max_min_sampling(
            data=dc_components, num_waypoints=k
        )
        dc_components["iix"] = np.arange(len(dc_components))
        waypoint_ix = dc_components.loc[waypoint_init]["iix"].values
        if self.verbose:
            print("Done.")

        return waypoint_ix

    def _get_greedy_centers(self, n_SEACells=None):
        """Initialize SEACells using fast greedy adaptive CSSP.

        From https://arxiv.org/pdf/1312.6838.pdf
        :param n_SEACells: (int) number of SEACells to initialize using greedy selection. If None specified,
                        all SEACells initialized using this method.
        :return: (np.ndarray) indices of cells to use as initial archetypes
        """
        K = self.K
        n = K.shape[0]

        if n_SEACells is None:
            k = self.k
        else:
            k = n_SEACells

        if self.verbose:
            print("Initializing residual matrix using greedy column selection")

        # precompute M.T * M
        # ATA = M.T @ M
        ATA = K

        if self.verbose:
            print("Initializing f and g...")

        f = np.array((ATA.multiply(ATA)).sum(axis=0)).ravel()
        # f = np.array((ATA * ATA).sum(axis=0)).ravel()
        g = np.array(ATA.diagonal()).ravel()

        d = np.zeros((k, n))
        omega = np.zeros((k, n))

        # keep track of selected indices
        centers = np.zeros(k, dtype=int)

        # sampling
        for j in tqdm(range(k)):
            score = f / g
            p = np.argmax(score)

            # print residuals
            np.sum(f)

            delta_term1 = ATA[:, p].toarray().squeeze()
            # print(delta_term1)
            delta_term2 = (
                np.multiply(omega[:, p].reshape(-1, 1), omega).sum(axis=0).squeeze()
            )
            delta = delta_term1 - delta_term2

            # some weird rounding errors
            delta[p] = np.max([0, delta[p]])

            o = delta / np.max([np.sqrt(delta[p]), 1e-6])
            omega_square_norm = np.linalg.norm(o) ** 2
            omega_hadamard = np.multiply(o, o)
            term1 = omega_square_norm * omega_hadamard

            # update f (term2)
            pl = np.zeros(n)
            for r in range(j):
                omega_r = omega[r, :]
                pl += np.dot(omega_r, o) * omega_r

            ATAo = (ATA @ o.reshape(-1, 1)).ravel()
            term2 = np.multiply(o, ATAo - pl)

            # update f
            f += -2.0 * term2 + term1

            # update g
            g += omega_hadamard

            # store omega and delta
            d[j, :] = delta
            omega[j, :] = o

            # add index
            centers[j] = int(p)

        return centers

    def _updateA(self, B, A_prev):
        """Update step for assigment matrix A.

        Given archetype matrix B and using kernel matrix K, compute assignment matrix A using constrained gradient
        descent via Frank-Wolfe algorithm.

        :param B: (n x k csr_matrix) defining SEACells as weighted combinations of cells
        :param A_prev: (n x k csr_matrix) defining previous weights used for assigning cells to SEACells
        :return: (n x k csr_matrix) defining updated weights used for assigning cells to SEACells
        """
        logger.info("inside updateA function")
        n, k = B.shape
        A = A_prev

        t = 0  # current iteration (determine multiplicative update)

        # precompute some gradient terms
        t2 = (self.K @ B).T
        t1 = t2 @ B

        # update rows of A for given number of iterations
        while t < self.max_FW_iter:
            #logger.info("current fw iter is")
            #logger.info(t)
            # compute gradient (must convert matrix to ndarray)
            G = 2.0 * np.array(t1 @ A - t2)

            # # get argmins - shape 1 x n
            amins = np.argmin(G, axis=0)
            amins = np.array(amins).reshape(-1)

            # # loop free implementation
            e = csr_matrix((np.ones(len(amins)), (amins, np.arange(n))), shape=A.shape)

            A += 2.0 / (t + 2.0) * (e - A)
            t += 1
        logger.info("done with updateA")
        return A

    def _updateB(self, A, B_prev):
        """Update step for archetype matrix B.

        Given assignment matrix A and using kernel matrix K, compute archetype matrix B using constrained gradient
        descent via Frank-Wolfe algorithm.

        :param A: (n x k csr_matrix) defining weights used for assigning cells to SEACells
        :param B_prev: (n x k csr_matrix) defining previous SEACells as weighted combinations of cells
        :return: (n x k csr_matrix) defining updated SEACells as weighted combinations of cells
        """
        logger.info("inside updateB function")
        K = self.K
        k, n = A.shape

        B = B_prev

        # keep track of error
        t = 0

        # precompute some terms
        t1 = A @ A.T
        t2 = K @ A.T

        # update rows of B for a given number of iterations
        while t < self.max_FW_iter:
            #logger.info("current fw iter is")
            #logger.info(t)
            # compute gradient (need to convert np.matrix to np.array)
            G = 2.0 * np.array(K @ B @ t1 - t2)

            # get all argmins
            amins = np.argmin(G, axis=0)
            amins = np.array(amins).reshape(-1)

            e = csr_matrix((np.ones(len(amins)), (amins, np.arange(k))), shape=B.shape)

            B += 2.0 / (t + 2.0) * (e - B)

            t += 1
        logger.info("done with updateB")
        return B

    def compute_reconstruction(self, A=None, B=None):
        """Compute reconstructed data matrix using learned archetypes (SEACells) and assignments.

        :param A: (k x n csr_matrix) defining weights used for assigning cells to SEACells
                If None provided, self.A is used.
        :param B: (n x k csr_matrix) defining SEACells as weighted combinations of cells
                If None provided, self.B is used.
        :return: (n x n csr_matrix) defining reconstructed data matrix.
        """
        if A is None:
            A = self.A_
        if B is None:
            B = self.B_

        if A is None or B is None:
            raise RuntimeError(
                "Either assignment matrix A or archetype matrix B is None."
            )
        return (self.kernel_matrix.dot(B)).dot(A)


    def compute_RSS(self, A=None, B=None, return_squared: bool = False):
        """
        Compute ||K - KBA||_F without forming n×n dense blocks.
        Uses identities:
        ||KBA||_F^2 = Tr( (A A^T) · ( (KB)^T (KB) ) )
        <K, KBA>    = Tr( A (K (K B)) ) = Tr( A U ),  where U = K (K B)
        Complexity ~ O(nnz(K) * k) instead of O(nnz(K) * n).
        """
        logger.info("inside compute_RSS (fast), i hope this time it works")

        # Defaults
        if A is None: A = self.A_
        if B is None: B = self.B_
        if A is None or B is None or self.kernel_matrix is None:
            raise RuntimeError("A, B, and kernel_matrix must be initialized before compute_RSS().")

        # Ensure sparse formats we can multiply efficiently
        K = self.kernel_matrix.tocsr()        # (n x n) CSR
        n = K.shape[0]
        #logger.info("compute_RSS: done sparse format K")

        # A: (k x n), B: (n x k)
        # Accept dense/sparse, but coerce types sensibly
        if sp.isspmatrix(A):
            A = A.tocsr().astype(np.float32, copy=False)
        else:
            A = np.asarray(A, dtype=np.float32, order="C")
        #logger.info("compute_RSS: done checking A")

        if sp.isspmatrix(B):
            B = B.tocsr().astype(np.float32, copy=False)
        else:
            B = np.asarray(B, dtype=np.float32, order="C")
        #logger.info("compute_RSS: done checking B")

        k = B.shape[1]
        assert B.shape == (n, k), f"B must be (n={n}, k={k})"
        assert A.shape == (k, n), f"A must be (k={k}, n={n})"

        # 1) ||K||_F^2 (sparse-safe)
        norm_K_sq = float(K.multiply(K).sum())
        #logger.info("compute_RSS: done norm_K_sq")

        # 2) W = K B   (n x k)    cost ~ O(nnz(K) * k)
        #    Use .dot to stay in SciPy sparse kernels
        W = K.dot(B) if sp.isspmatrix(B) else K.dot(B)
        #logger.info("compute_RSS: done W=K.dot(B)")

        # 3) ||KBA||_F^2 = Tr( (A A^T) * (W^T W) )   all small (k x k)
        #    T = A A^T  (k x k);  S = W^T W  (k x k).  Then ||KBA||^2 = sum(T * S)
        #    Build T and S as float64 for numeric stability.
        if sp.isspmatrix(A):
            #T = (A @ A.T).astype(np.float64)
            #T = T.toarray() if sp.isspmatrix(T) else np.asarray(T)
            T = (A @ A.T).toarray().astype(np.float64)
        else:
            T = (A @ A.T).astype(np.float64)
        #logger.info("compute_RSS: done T")

        S = (W.T @ W).astype(np.float64)  # (k x k), dense
        #logger.info("compute_RSS: done S")
        norm_R_sq = float((T * S).sum())
        #logger.info("compute_RSS: done norm_R_sq")

        # 4) inner = <K, KBA> = Tr( A U ), where U = K (K B) = K W   (n x k)
        U = K.dot(W) # (n x k)
        #logger.info(f"compute_RSS shapes: K={K.shape}, B={B.shape}, A={A.shape}, W={W.shape}, U={U.shape}")


        if sp.isspmatrix(A):
            #inner = float(np.trace((A @ U).astype(np.float64)))   # A(k x n) @ U(n x k) -> (k x k)
            inner = float(A.multiply(U.T).sum()) # A is (k x n), U.T is (k x n); elementwise multiply, then sum.
        else:
            #logger.info("inner else path")
            inner = float(np.einsum('ij,ij->', A, U.T, dtype=np.float64)) # dense path; einsum is stable and shape-agnostic
            #inner = float(np.trace(A @ U))                         # dense path
        
        #logger.info("compute_RSS: done inner")

        # 5) RSS^2 = ||K||_F^2 + ||KBA||_F^2 - 2 * inner
        rss_sq = norm_K_sq + norm_R_sq - 2.0 * inner
        if rss_sq < 0.0:  # numerical guard
            rss_sq = max(rss_sq, 0.0)
        #logger.info("compute_RSS: done rss_sq")

        logger.info("compute_RSS (fast) done")
        return float(rss_sq) if return_squared else float(np.sqrt(np.float64(rss_sq)))


    def compute_RSS_previous2(self, A=None, B=None, block: int = 64, return_squared: bool = False):
        """
        Compute ||K - KBA||_F (default) without forming the n×n reconstruction.
        If return_squared=True, return the squared Frobenius norm.

        K: self.kernel_matrix (n×n, CSR)
        B: (n×k) (dense or sparse)
        A: (k×n) (dense or sparse)
        """
        logger.info("inside compute_RSS function, lets see if it pops out errors")

        # Defaults
        if A is None:
            A = self.A_
        if B is None:
            B = self.B_

        # Ensure formats/dtypes we can multiply efficiently
        K = self.kernel_matrix.tocsr()
        n = K.shape[0]

        # Use float32 for intermediates unless float64 is required
        if sp.isspmatrix(A):
            A = A.astype(np.float32, copy=False)
        else:
            A = np.asarray(A, dtype=np.float32, order="C")

        if sp.isspmatrix(B):
            B = B.astype(np.float32, copy=False).tocsr()
        else:
            B = np.asarray(B, dtype=np.float32, order="C")

        # 1) ||K||_F^2
        norm_K_sq = float(K.power(2).sum())

        # 2) Accumulate ||KBA||_F^2 and <K, KBA> in column blocks of A
        norm_R_sq = 0.0
        inner = 0.0

        indptr = K.indptr
        indices = K.indices
        data = K.data

        # Process A in blocks of columns to bound memory
        for j0 in range(0, A.shape[1], block):
            j1 = min(j0 + block, A.shape[1])
            b = j1 - j0  # block width

            # A_blk: (k × b) dense
            if sp.isspmatrix(A):
                A_blk = A[:, j0:j1].toarray(order="C")
            else:
                A_blk = A[:, j0:j1]  # already dense float32

            # V = B @ A_blk  -> (n × b) dense
            if sp.isspmatrix(B):
                V = B.dot(A_blk)
            else:
                V = B @ A_blk
            V = np.asarray(V, dtype=np.float32, order="C")

            # R_blk = K @ V   -> (n × b) dense (SpMM)
            R_blk = K.dot(V)  # CSR × dense → dense (n × b, cols correspond to j0..j1-1)

            # 2a) accumulate ||R_blk||_F^2
            norm_R_sq += float((R_blk * R_blk).sum())

            # 2b) accumulate <K, R_blk> only at K's nonzeros but restricted to this block:
            #     sum_i sum_{j in nz(K[i,:]) ∩ [j0, j1)} K[i,j] * R_blk[i, j - j0]
            for i in range(n):
                start, end = indptr[i], indptr[i + 1]
                if start == end:
                    continue

                row_cols = indices[start:end]   # global column indices of K's nonzeros in row i
                row_vals = data[start:end]      # corresponding values

                # keep only those that land in the current block [j0, j1)
                mask = (row_cols >= j0) & (row_cols < j1)
                if not np.any(mask):
                    continue

                cols_local = row_cols[mask] - j0   # shift to 0..(b-1)
                vals_local = row_vals[mask]
                inner += float((vals_local * R_blk[i, cols_local]).sum())

            # (optional) free block temporaries early
            del A_blk, V, R_blk

        # ||K - R||_F^2 = ||K||_F^2 + ||R||_F^2 - 2 <K, R>
        rss_sq = norm_K_sq + norm_R_sq - 2.0 * inner

        # Numerical guard (rare negatives from roundoff)
        if rss_sq < 0.0:
            rss_sq = max(rss_sq, 0.0)
        
        logger.info("compute_RSS safely done!")

        if return_squared:
            return rss_sq
        else:
            return float(np.sqrt(np.float64(rss_sq)))



    def compute_RSS_previous1(self, A=None, B=None, block: int = 64, return_squared: bool = False):
        """
        Compute ||K - KBA||_F (default) without forming the n×n reconstruction.
        If return_squared=True, return the squared Frobenius norm.

        K: self.kernel_matrix (n×n, CSR)
        B: (n×k) (dense or sparse)
        A: (k×n) (dense or sparse)
        """
        logger.info("inside compute_RSS function, lets see if it pops out errors")
        if A is None:
            A = self.A_
        if B is None:
            B = self.B_

        # Ensure sparse/dense formats we can multiply efficiently
        K = self.kernel_matrix.tocsr()
        n = K.shape[0]

        # Use float32 to reduce memory unless you require float64 throughout
        if sp.isspmatrix(A):
            A = A.astype(np.float32, copy=False)
        else:
            A = np.asarray(A, dtype=np.float32, order="C")

        if sp.isspmatrix(B):
            B = B.astype(np.float32, copy=False).tocsr()
        else:
            B = np.asarray(B, dtype=np.float32, order="C")

        # 1) ||K||_F^2
        norm_K_sq = float(K.power(2).sum())

        # 2) Accumulate ||KBA||_F^2 and <K, KBA> in column blocks of A
        norm_R_sq = 0.0
        inner = 0.0

        indptr = K.indptr
        indices = K.indices
        data = K.data

        # Process A in blocks of columns to bound memory
        for j0 in range(0, A.shape[1], block):
            j1 = min(j0 + block, A.shape[1])

            # A_blk: (k × b) dense
            if sp.isspmatrix(A):
                A_blk = A[:, j0:j1].toarray(order="C")
            else:
                A_blk = A[:, j0:j1]  # already dense

            # V = B @ A_blk  -> (n × b) dense
            if sp.isspmatrix(B):
                V = B.dot(A_blk)
            else:
                V = B @ A_blk
            V = np.asarray(V, dtype=np.float32, order="C")

            # R_blk = K @ V   -> (n × b) dense (SpMM)
            R_blk = K.dot(V)  # CSR × dense

            # 2a) accumulate ||R_blk||_F^2
            norm_R_sq += float(np.square(R_blk, dtype=np.float32).sum())

            # 2b) accumulate <K, R_blk> only at K's nonzeros:
            #     sum_i sum_{j in nz(K[i,:])} K[i,j] * R_blk[i, j]
            for i in range(n):
                start, end = indptr[i], indptr[i + 1]
                cols = indices[start:end]
                if cols.size:
                    inner += float((data[start:end] * R_blk[i, cols]).sum())

            # (optional) free block temporaries early
            del A_blk, V, R_blk

        # ||K - R||_F^2 = ||K||_F^2 + ||R||_F^2 - 2 <K, R>
        rss_sq = norm_K_sq + norm_R_sq - 2.0 * inner

        # Numerical guard (rare negatives from roundoff)
        if rss_sq < 0:
            rss_sq = max(rss_sq, 0.0)

        if return_squared:
            return rss_sq
        else:
            return float(np.sqrt(rss_sq, dtype=np.float64)) 


    def compute_RSS_previous(self, A=None, B=None):
        """Compute residual sum of squares error in difference between reconstruction and true data matrix.

        :param A: (k x n csr_matrix) defining weights used for assigning cells to SEACells
                If None provided, self.A is used.
        :param B: (n x k csr_matrix) defining SEACells as weighted combinations of cells
                If None provided, self.B is used.
        :return:
            ||X-XBA||^2 - (float) square difference between true data and reconstruction.
        """
        if A is None:
            A = self.A_
        if B is None:
            B = self.B_

        reconstruction = self.compute_reconstruction(A, B)
        return norm(self.kernel_matrix - reconstruction)

    def plot_convergence(self, save_as=None, show=True):
        """Plot behaviour of squared error over iterations.

        :param save_as: (str) name of file which figure is saved as. If None, no plot is saved.
        :param show: (bool) whether to show plot
        :return: None.
        """
        import matplotlib.pyplot as plt

        plt.figure()
        plt.plot(self.RSS_iters)
        plt.title("Reconstruction Error over Iterations")
        plt.xlabel("Iterations")
        plt.ylabel("Squared Error")
        if save_as is not None:
            plt.savefig(save_as, dpi=150)
        if show:
            plt.show()
        plt.close()

    def step(self):
        """Perform one iteration of SEACell algorithm. Update assignment matrix A and archetype matrix B.

        :return: None.
        """
        A = self.A_
        B = self.B_

        if self.K is None:
            raise RuntimeError(
                "Kernel matrix has not been computed. Run model.construct_kernel_matrix() first."
            )

        if A is None:
            raise RuntimeError(
                "Cell to SEACell assignment matrix has not been initialised. Run model.initialize() first."
            )

        if B is None:
            raise RuntimeError(
                "Archetype matrix has not been initialised. Run model.initialize() first."
            )

        A = self._updateA(B, A)
        B = self._updateB(A, B)

        self.RSS_iters.append(self.compute_RSS(A, B))

        self.A_ = A
        self.B_ = B

        # Label cells by SEACells assignment
        labels = self.get_hard_assignments()
        self.ad.obs["SEACell"] = labels["SEACell"]

        return

    def _fit(
        self,
        max_iter: int = 50,
        min_iter: int = 10,
        initial_archetypes=None,
        initial_assignments=None,
    ):
        """Internal method to compute archetypes and loadings given kernel matrix K.

        Iteratively updates A and B matrices until maximum number of iterations or convergence has been achieved.

        Modifies ad.obs in place to add 'SEACell' labels to cells.
        :param max_iter: (int) maximum number of iterations to perform
        :param min_iter: (int) minimum number of iterations to perform
        :param initial_archetypes: (array) initial archetypes to use. If None, random initialisation is used.
        :param initial_assignments: (array) initial assignments to use. If None, random initialisation is used.
        :return: None
        """
        logger.info("we are inside _fit function")
        
        self.initialize(
            initial_archetypes=initial_archetypes,
            initial_assignments=initial_assignments,
        )
        
        converged = False
        n_iter = 0
        
        while (not converged and n_iter < max_iter) or n_iter < min_iter:
            n_iter += 1
            logger.info("current iter is")
            logger.info(n_iter)
            if n_iter == 1 or (n_iter) % 10 == 0:
                #logger.info("current iter is")
                #logger.info(n_iter)
                if self.verbose:
                    print(f"Starting iteration {n_iter}.")

            self.step()
            
            logger.info("completed current iter is")
            logger.info(n_iter)
            
            if n_iter == 1 or (n_iter) % 10 == 0:
                #logger.info("completed this iter")
                if self.verbose:
                    print(f"Completed iteration {n_iter}.")

            # Check for convergence
            if (
                np.abs(self.RSS_iters[-2] - self.RSS_iters[-1])
                < self.convergence_threshold
            ):
                if self.verbose:
                    print(f"Converged after {n_iter} iterations.")
                converged = True
                logger.info("converged!! after this many iters")
                logger.info(n_iter)

        self.Z_ = self.B_.T @ self.K

        # Label cells by SEACells assignment
        labels = self.get_hard_assignments()
        self.ad.obs["SEACell"] = labels["SEACell"]

        if not converged:
            raise RuntimeWarning(
                "Warning: Algorithm has not converged - you may need to increase the maximum number of iterations"
            )
        logger.info("done with this _fit function")
        return

    def fit(
        self,
        max_iter: int = 100,
        min_iter: int = 10,
        initial_archetypes=None,
        initial_assignments=None,
    ):
        """Compute archetypes and loadings given kernel matrix K.

        Iteratively updates A and B matrices until maximum number of iterations or convergence has been achieved.
        :param max_iter: (int) maximum number of iterations to perform (default 100)
        :param min_iter: (int) minimum number of iterations to perform (default 10)
        :param initial_archetypes: (array) initial archetypes to use. If None, random initialisation is used.
        :param initial_assignments: (array) initial assignments to use. If None, random initialisation is used.
        :return: None.
        """
        if max_iter < min_iter:
            raise ValueError(
                "The maximum number of iterations specified is lower than the minimum number of iterations specified."
            )
        self._fit(
            max_iter=max_iter,
            min_iter=min_iter,
            initial_archetypes=initial_archetypes,
            initial_assignments=initial_assignments,
        )

    def get_archetype_matrix(self):
        """Return k x n matrix of archetypes computed as the product of the archetype matrix B and the kernel matrix K."""
        return self.Z_

    def get_soft_assignments(self):
        """Return soft SEACell assignment.

        Returns a tuple of (labels, weights) where labels is a dataframe with SEACell assignments for the top 5
        SEACell assignments for each cell and weights is an array with the corresponding weights for each assignment.
        :return: (pd.DataFrame, np.array) with labels and weights.
        """
        archetype_labels = self.get_hard_archetypes()
        A = copy.deepcopy(self.A_.T)

        labels = []
        weights = []
        for _i in range(5):
            l = A.argmax(1)
            labels.append(archetype_labels[l])
            weights.append(A[np.arange(A.shape[0]), l])
            A[np.arange(A.shape[0]), l] = -1

        weights = np.vstack(weights).T
        labels = np.vstack(labels).T

        soft_labels = pd.DataFrame(labels)
        soft_labels.index = self.ad.obs_names

        return soft_labels, weights

    def get_hard_assignments(self):
        """Returns a dataframe with the SEACell assignment for each cell.

        The assignment is the SEACell with the highest assignment weight.
        :return: (pd.DataFrame) with SEACell assignments.
        """
        # Use argmax to get the index with the highest assignment weight
        assmts = np.array(self.A_.argmax(0)).reshape(-1)

        df = pd.DataFrame({"SEACell": [f"SEACell-{i}" for i in assmts]})
        df.index = self.ad.obs_names
        df.index.name = "index"
        return df

    def get_hard_archetypes(self):
        """Return the names of cells most strongly identified as archetypes.

        :return list of archetype names.
        """
        return self.ad.obs_names[self.B_.argmax(0)]

    def save_model(self, outdir):
        """Save the model to a pickle file.

        :param outdir: (str) path to directory to save to
        :return: None.
        """
        import pickle

        with open(outdir + "/model.pkl", "wb") as f:
            pickle.dump(self, f)
        return None

    def save_assignments(self, outdir):
        """Save SEACells assignment.

        Saves:
        (1) the cell to SEACell assignments to a csv file with the name 'SEACells.csv'.
        (2) the kernel matrix to a .npz file with the name 'kernel_matrix.npz'.
        (3) the archetype matrix to a .npz file with the name 'A.npz'.
        (4) the loading matrix to a .npz file with the name 'B.npz'.

        :param outdir: (str) path to directory to save to
        :return: None
        """
        import os

        os.makedirs(outdir, exist_ok=True)
        save_npz(outdir + "/kernel_matrix.npz", self.kernel_matrix)
        save_npz(outdir + "/A.npz", self.A_.T)
        save_npz(outdir + "/B.npz", self.B_)

        labels = self.get_hard_assignments()
        labels.to_csv(outdir + "/SEACells.csv")
        return None
