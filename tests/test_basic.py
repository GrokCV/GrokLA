import unittest
from pymatrix import Matrix

class TestBasic(unittest.TestCase):
    def test_det(self):
        A = Matrix([[1, 2], [3, 4]])
        self.assertAlmostEqual(A.det(), -2.0, places=6)

    def test_inv(self):
        A = Matrix([[1, 2], [3, 4]])
        invA = A.inv()
        I = A @ invA
        self.assertAlmostEqual(I._data[0][0], 1.0, places=5)
        self.assertAlmostEqual(I._data[1][1], 1.0, places=5)
        self.assertAlmostEqual(I._data[0][1], 0.0, places=5)
        self.assertAlmostEqual(I._data[1][0], 0.0, places=5)

if __name__ == "__main__":
    unittest.main()
