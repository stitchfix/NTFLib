import unittest
import numpy as np
from ntflib import utils


class TestUtils(unittest.TestCase):
    def test_top_sparse3(self):
        shape, rank, k, factors, x, x_indices, x_vals = \
                utils.generate_dataset()
        model = utils.parafac(factors)
        for beta in [1.0, 2.0, 1.5]:
            for factor in range(3):
                # Generate the top numerator for the reference dense method
                einstr = utils.generate_dense(rank, factor)
                # Get all factors that aren't the current factor
                mode_factors = [a for j, a in enumerate(factors) if j != factor]
                mode_factors += [x * (model ** (beta - 2.)), ]
                top_d = np.einsum(einstr, *mode_factors)

                # Now get the top numerator for the sparse method
                top_s = np.zeros(factors[factor].shape, dtype=np.float32)
                utils.top_sparse3(x_indices, x_vals, top_s, beta, factor, *factors)

                # Now get the top numerator for the sparse method using numba
                top_n = np.zeros(factors[factor].shape, dtype=np.float32)
                utils.top_sparse3_numba(x_indices, x_vals, top_n, beta, factor, *factors)

                print(top_d, top_s, top_n)
                print(top_d - top_n)
                print(np.histogram(np.abs((top_d - top_n) / top_d)))
                result = np.allclose(top_d, top_s, rtol=1e-2, atol=1e-2)
                self.assertTrue(result)
                result = np.allclose(top_d, top_n, rtol=1e-2, atol=1e-2)
                self.assertTrue(result)


    def test_bot_sparse3(self):
        shape, rank, k, factors, x, x_indices, x_vals = \
                utils.generate_dataset()
        model = utils.parafac(factors)
        for beta in [1.0, 2.0, 1.5]:
            for factor in range(3):
                # Generate the bottom denominator for the reference dense method
                einstr = utils.generate_dense(rank, factor)
                # Get all factors that aren't the current factor
                mode_factors = [a for j, a in enumerate(factors) if j != factor]
                mode_factors += [(model ** (beta - 1.)), ]
                bot_d = np.einsum(einstr, *mode_factors)

                # Now get the bottom denominator for the sparse method
                bot_s = np.zeros(factors[factor].shape, dtype=np.float32)
                utils.bot_sparse3(x_indices, x_vals, bot_s, beta, factor, *factors)

                # Now get the bottom denominator for the sparse method
                bot_n = np.zeros(factors[factor].shape, dtype=np.float32)
                utils.bot_sparse3(x_indices, x_vals, bot_n, beta, factor, *factors)

                result = np.allclose(bot_d, bot_s, rtol=1e-2)
                self.assertTrue(result)
                result = np.allclose(bot_d, bot_n, rtol=1e-2)
                self.assertTrue(result)


    def test_beta_divergence(self):
        shape, rank, k, factors, x, x_indices, x_vals = \
                utils.generate_dataset()
        for beta in [1, 1.5, 2]:
            a = x
            b = utils.parafac(factors)
            div_d = utils.beta_divergence_dense(a, b, beta)
            div_s = utils.beta_divergence(x_indices, x_vals, beta, *factors)
            div_s.shape = div_d.shape
            close = np.allclose(div_s, div_d, rtol=1e-5)
            self.assertTrue(close)
