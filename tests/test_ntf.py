import unittest
from ntf import BetaNTF

class TestNTFLib(unittest.TestCase):
    def test_init(self):
        shape = (10, 6)
        bnf = BetaNTF(shape)
        for s in shape:
            self.assertTrue(len(bnf._factors[i]), s)

    def test_fit(self):
        shape = (10, 6)
        bnf = BetaNTF(shape)
        pass
