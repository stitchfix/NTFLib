import unittest
import utils
import numpy as np


def generate_dense(rank, mode):
    # will be : 'az,cz,abc->bz'
    # when mode=1
    request = ''
    for r in range(rank):
        if r == mode: continue
        request += utils.alpha[r] + 'z,'
    request += utils.alphabet[:rank] + '->'
    request += utils.alphabet[mode] + 'z'
    return request


def generate_dataset():
    shape = (4, 5, 6)
    rank = len(shape)
    k = 2
    init = [np.abs(np.random.randn((s, k))).astype(np.float32) for s in shape]
    hidn = [np.abs(np.random.randn((s, k))).astype(np.float32) for s in shape]
    x = utils.parafac(hidn)
    x_indices = np.indices(x.shape)
    x_vals = x.ravel()
    return shape, rank, k, init, x, x_indices, x_vals


class TestUtils(unittest.TestCase):
    def test_top_sparse3(self):
        mode = 0
        shape, rank, k, factors, x, x_indices, x_vals = generate_dataset()
        # Get all factors that aren't the current factor
        mode_factors = [a for j, a in enumerate(factors) if j != mode]

        # Now get the top numerator for the sparse method
        model = utils.parafac(factors)
        top_s = np.zeros(factors[mode].shape, dtype=np.float32)
        utils.top_sparse3(x_indices, x_vals, model, out, beta, mode_factors[0],
                          mode_factors[1])

        # Generate the top numerator for the reference dense method
        einstr = generate_dense(rank, mode)
        mode_factors += [X * (model ** (beta - 2.)), ]
        top_d = np.einsum(einstr, *mode_factors)

        result = np.allclose(top_d, top_s, rtol=1e-2)
        self.assertTrue(result)
