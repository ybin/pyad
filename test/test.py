import unittest
import numpy as np
from .. import ad


class MyTestCase(unittest.TestCase):
    def test_add(self):
        """
        c = a + b
        """
        _a = ad.variable([1, 1], name='a')
        _b = ad.constant([2, 2], name='b')
        _c = ad.add(_a, _b, 'c')

        # initialize grad to 1's
        _c.grad = np.ones_like(_c.value)
        _c.backward()

        self.assertTrue(all(_a.grad == [1, 1]))
        self.assertEqual(_b.grad, 0)

    def test_sub(self):
        """
        c = a - b
        """
        _a = ad.variable([1, 1], name='a')
        _b = ad.variable([2, 2], name='b')
        _c = ad.sub(_a, _b, 'c')

        # initialize grad to 1's
        _c.grad = np.ones_like(_c.value)
        _c.backward()

        self.assertTrue(all(_a.grad == [1, 1]))
        self.assertTrue(all(_b.grad == [-1, -1]))

    def test_mul(self):
        """
        c = a * b
        """
        _a = ad.variable([1, 1], name='a')
        _b = ad.variable([2, 2], name='b')
        _c = ad.mul(_a, _b, 'c')

        # initialize grad to 1's
        _c.grad = np.ones_like(_c.value)
        _c.backward()

        self.assertTrue(all(_a.grad == _b.value))
        self.assertTrue(all(_b.grad == _a.value))

    def test_div(self):
        """
        c = a / b
        """
        _a = ad.variable([1, 1], name='a')
        _b = ad.variable([2, 2], name='b')
        _c = ad.div(_a, _b, 'c')

        # initialize grad to 1's
        _c.grad = np.ones_like(_c.value)
        _c.backward()

        self.assertTrue(all(_a.grad == 1 / _b.value))
        self.assertTrue(all(_b.grad == -1 * _a.value))

    def test_graph(self):
        """
        e = (a + b) * (b + 1)
        """
        _a = ad.variable(2, name='a')
        _b = ad.variable(1, name='b')
        _one = ad.constant(1, name='one')
        _c = ad.add(_a, _b, 'c')
        _d = ad.add(_b, _one)
        _e = ad.mul(_c, _d, name='e')

        # initialize grad to 1's
        _e.grad = np.ones_like(_c.value)
        _e.backward()

        self.assertTrue(_a.grad == 2)
        self.assertTrue(_b.grad == 5)
        pass


if __name__ == '__main__':
    unittest.main()
