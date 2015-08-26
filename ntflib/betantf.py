import numpy as np
import warnings
from . import utils

# This is a form of Non-negative Tensor Factorization
# that minimizes the beta-divergence. Typical NTF specializes
# in factorizing dense tensors, but we are keen to use sparse
# data and keep memory pressure low.


class BetaNTF():
    def __init__(self, shape, n_components=5, beta=1, n_iters=10,
                 verbose=True):
        """
        BetaNTF(shape, n_components=5, beta=1, n_iters=10, verbose=True)

        shape: int (n_dimensions)
        `shape` should have one entry for the number of unique elements
        in each input dimension of the tensor. 

        n_components: int  (default=5)
        Number of components to use in the factorization of the input tensor.
        The more components the better the fit, but (crudley speaking) the less
        general your solution, and the more time it will take to compute.

        beta: float (default=1)
        The cost function is parameterized by beta such that beta=2 is
        equivalent to Euclidean distance, beta=1 is the KL divergence, and
        beta=0 Itakura-Saito divergence. 

        """
        self.shape = shape
        self.n_components = n_components
        self.beta = np.float(beta)
        self.n_iters = n_iters
        self.verbose = verbose
        self.rank = len(shape)
        fact = lambda s, n: np.abs(np.random.randn(s, n)).astype(np.float32)
        self._factors = [fact(s, n_components) for s in shape]
        self.topf = utils.tops[len(shape)]
        self.botf = utils.bots[len(shape)]

    def _check_input(self, x_indices, x_vals):
        """ Check that every marginal is defined.
        E.g., for every dimension we have at least one
        observation. This is crucial to the factor updates
        we cannot tolerate a whole dimension with no data."""
        for col in range(x_indices.shape[1]):
            rank = x_indices[:, col]
            msg = "Rank did not match shape; is column %i "
            msg += "starting with zero and strictly contiguous integers?"
            msg = msg % col
            if rank.max() + 1 != np.unique(rank).shape[0]:
                warnings.warn(msg)
        assert len(x_vals) == len(x_indices)
        assert np.all(np.isfinite(x_vals))
        assert np.all(x_vals >= 0)

    def fit(self, x_indices, x_vals):
        """ 
        fit(x_indices, x_vals)

        Fit the tensor factors to a sparse tensor.
        The coordinates of the non-missing values are given 
        by `x_indices` and the values at those points are
        give by `x_vals`.

        For example, if a rating of 3.0 is given by user_id=3, movie=10
        on date_id=30, then construct these arrays such that 
        x_indices[0] = [3, 10, 30] and x_vals[0]=3.0. 

        Note that the indices must be monotonic and contiguous --
        there can be no missing "gaps" in the indices so that
        np.unique(index) == index.max(). The indices must start at
        zero.

        Parameters
        ----------
        x_indices: np.int32 (n_row, n_dim) 
        Coordinates at which `x_vals` occur. This is the
        "address" of a cell in a tensor.

        x_vals: np.float32 (n_row) 
        The value of a cell in a tensor, e.g. the output 
        rating of a user or item.
        """
        eps = 1e-8
        # Reduce the cost in each iteration
        x_indices = x_indices.astype(np.int32)
        x_vals = x_vals.astype(np.float32)
        self.x_indices = x_indices
        self.x_vals = x_vals
        self._check_input(x_indices, x_vals)
        for it in range(self.n_iters):
            # Update each factor individually
            for factor in range(self.rank):
                # Get all factors that aren't the current factor
                fctrs = [a for j, a in enumerate(self._factors) if j != factor]
                assert len(fctrs) == self.rank - 1
                # Get the numerator for the update multiplier
                top = np.zeros(self._factors[factor].shape, dtype=np.float32)
                bot = np.zeros(self._factors[factor].shape, dtype=np.float32)
                self.topf(x_indices, x_vals, top, self.beta, factor,
                          *self._factors)
                self.botf(x_indices, x_vals, bot, self.beta, factor,
                          *self._factors)
                frac = (eps + top) / (eps + bot)
                assert np.all(np.isfinite(frac))
                self._factors[factor] *= frac
                assert np.all(np.isfinite(self._factors[factor]))
                self.log(it, factor)
        return self.impute(x_indices)

    def score(self, x_indices=None, x_vals=None):
        if x_indices is None:
            x_indices = self.x_indices
        if x_vals is None:
            x_vals = self.x_vals
        score = utils.beta_divergence(x_indices, x_vals, self.beta,
                                      *self._factors)
        return score.sum()

    def impute(self, x_indices):
        x_vals = np.zeros(x_indices.shape[0])
        utils.parafac_sparse(x_indices, x_vals, *self._factors)
        return x_vals

    def log(self, it, factor, score=None):
        if score is None:
            score = self.score()
        if self.verbose:
            msg = "Update Iter %i Factor %i" % (it, factor)
            msg += " Score %1.1f" % score
            print(msg)
