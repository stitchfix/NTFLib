import unittest
import utils
from ntf import BetaNTF

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
        reconstructed = bnf.fit(x_indices, x_vals)
        after = bnf.score()
        self.assertTrue(after < before)
