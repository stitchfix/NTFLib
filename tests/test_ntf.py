import unittest
from ntf import BetaNTF

class TestNTFLib(unittest.TestCase):
    def test_init(self):
        shape = (10, 6, 2)
        bnf = BetaNTF(shape)
        for i, s in enumerate(shape):
            self.assertTrue(bnf._factors[i].shape[0], s)
