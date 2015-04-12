import unittest
import utils
import numpy as np


def generate_dense(rank, mode):
    # will be : 'az,cz,abc->bz'
    # when mode=1
    request = ''
    for r in range(rank):
        if r == mode: continue
        request += utils.alphabet[r] + 'z,'
    request += utils.alphabet[:rank] + '->'
    request += utils.alphabet[mode] + 'z'
    return request

def gen_rand(s, k):
    d = np.abs(np.random.randn(s * k)).reshape((s, k))
    return d.astype(np.float32)

def generate_dataset():
    shape = (4, 5, 6)
    rank = len(shape)
    k = 2
    init = [gen_rand(s, k) for s in shape]
    hidden = [gen_rand(s, k) for s in shape]
    x = utils.parafac(hidden)
    x_indices = np.array([a.ravel() for a in np.indices(shape)]).T
    x_vals = x.ravel()
    return shape, rank, k, init, x, x_indices, x_vals


class TestUtils(unittest.TestCase):
    def test_top_sparse3(self):
        factor = 0
        beta = 1
        shape, rank, k, factors, x, x_indices, x_vals = generate_dataset()
        model = utils.parafac(factors)

        # Generate the top numerator for the reference dense method
        einstr = generate_dense(rank, factor)
        # Get all factors that aren't the current factor
        mode_factors = [a for j, a in enumerate(factors) if j != factor]
        mode_factors += [x * (model ** (beta - 2.)), ]
        top_d = np.einsum(einstr, *mode_factors)

        # Now get the top numerator for the sparse method
        top_s = np.zeros(factors[factor].shape, dtype=np.float32)
        utils.top_sparse3(x_indices, x_vals, top_s, beta, factor, *factors)

        result = np.allclose(top_d, top_s, rtol=1e-5)
        self.assertTrue(result)
