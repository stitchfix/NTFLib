import numpy as np
import numba
import utils

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
        self.top_sparse = utils.tops[len(shape)]
        self.bot_sparse = utils.bots[len(shape)]

    def fit(self, x_indices, x_vals):
        eps = 1e-8
        # Reduce the cost in each iteration
        x_indices = x_indices.astype(np.float32)
        x_vals = x_vals.astype(np.float32)
        for it in range(self.n_iters):
            # Update each factor individually
            for factor in range(self.rank):
                # Get current model
                model = utils.parafac(self._factors)
                # Get all factors that aren't the current factor
                fctrs = [a for j, a in enumerate(self._factors) if j != factor]
                # Get the numerator for the update multiplier
                top = np.zeros(self.shape, dtype=np.float32)
                bot = np.zeros(self.shape, dtype=np.float32)
                self.top_sparse(x_indices, x_vals, model, top, *fctrs)
                self.bot_sparse(x_indices, x_vals, model, bot, *fctrs)
                self._factors[factor] *= (eps + top) / (eps + bot)
                score = utils.beta_divergence(x_indices, x_vals, model,
                                              self.beta)
                self.log(it, factor, score=score)

    def log(self, it, factor, score=None):
        if self.verbose:
            msg = "Update Iter %i Factor %i" % (it, factor)
            if score is not None:
                msg += " Score %1.1f" % score
            print(msg)
