import unittest
import utils
import numpy as np
from betantf import BetaNTF

def rmse(x, y):
    return np.sqrt((x - y)**2.0).sum()

class TestNTFLib(unittest.TestCase):
    def test_init(self):
        shape = (10, 6, 2)
        bnf = BetaNTF(shape)
        for i, s in enumerate(shape):
            self.assertTrue(bnf._factors[i].shape[0], s)

    def test_fit(self):
        shape, rank, k, factors, x, x_indices, x_vals = \
                utils.generate_dataset()
        bnf = BetaNTF(shape, n_components=k)
        before = bnf.score(x_indices, x_vals)
        initial = bnf.impute()
        reconstructed = bnf.fit(x_indices, x_vals)
        after = bnf.score()
        self.assertTrue(after < before)
        e_0 = rmse(x, initial)
        e_1 = rmse(x, reconstructed)
        self.assertTrue(e_0 > e_1)
