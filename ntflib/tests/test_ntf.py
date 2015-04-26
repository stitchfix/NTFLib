import unittest
import numpy as np
from ntflib import betantf
from ntflib import utils

def rmse(x, y):
    return np.sqrt((x - y)**2.0).sum()

class TestNTFLib(unittest.TestCase):
    def test_init(self):
        shape = (10, 6, 2)
        bnf = betantf.BetaNTF(shape)
        for i, s in enumerate(shape):
            self.assertTrue(bnf._factors[i].shape[0], s)

    def test_fit(self):
        shape, rank, k, factors, x, x_indices, x_vals = \
                utils.generate_dataset()
        bnf = betantf.BetaNTF(shape, n_components=k)
        before = bnf.score(x_indices, x_vals)
        initial = bnf.impute(x_indices)
        initial.shape = x.shape
        reconstructed = bnf.fit(x_indices, x_vals)
        reconstructed.shape = x.shape
        after = bnf.score()
        self.assertTrue(after < before)
        e_0 = rmse(x, initial)
        e_1 = rmse(x, reconstructed)
        self.assertTrue(e_0 > e_1)
