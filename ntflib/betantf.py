import numpy as np
from . import utils

# This is a form of Non-negative Tensor Factorization
# that minimizes the beta-divergence. Typical NTF specializes
# in factorizing dense tensors, but we are keen to use sparse
# data and keep memory pressure low.


class BetaNTF():
    def __init__(self, shape, n_components=5, beta=1, n_iters=10,
                 verbose=True):
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
            assert rank.max() + 1 == np.unique(rank).shape[0]
        assert len(x_vals) == len(x_indices)

    def fit(self, x_indices, x_vals):
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
                # Get current model
                model = utils.parafac(self._factors)
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
                self._factors[factor] *= (eps + top) / (eps + bot)
                assert np.all(np.isfinite(self._factors[factor]))
                self.log(it, factor)
        return self.impute()

    def score(self, x_indices=None, x_vals=None):
        if x_indices is None:
            x_indices = self.x_indices
        if x_vals is None:
            x_vals = self.x_vals
        model = utils.parafac(self._factors)
        score = utils.beta_divergence(x_indices, x_vals, model,
                                      self.beta)
        return score.sum()

    def impute(self):
        return utils.parafac(self._factors)

    def log(self, it, factor, score=None):
        if score is None:
            score = self.score()
        if self.verbose:
            msg = "Update Iter %i Factor %i" % (it, factor)
            msg += " Score %1.1f" % score
            print(msg)
