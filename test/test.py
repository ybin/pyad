import unittest
import numpy as np
from .. import ad


class MyTestCase(unittest.TestCase):
    def test_add(self):
        _a = ad.variable(np.array([1, 1]), name='a')
        _b = ad.constant(np.array([2, 2]), name='b')
        _c = ad.add(_a, _b, 'c')

        # initialize grad to 1's
        _c.grad = np.ones_like(_c.value)
        _c.backward()

        self.assertTrue(all(_a.grad == [1, 1]))
        self.assertEqual(_b.grad, 0)

    def test_sub(self):
        _a = ad.variable(np.array([1, 1]), name='a')
        _b = ad.variable(np.array([2, 2]), name='b')
        _c = ad.sub(_a, _b, 'c')

        # initialize grad to 1's
        _c.grad = np.ones_like(_c.value)
        _c.backward()

        self.assertTrue(all(_a.grad == [1, 1]))
        self.assertTrue(all(_b.grad == [-1, -1]))

    def test_graph(self):
        pass


if __name__ == '__main__':
    unittest.main()
