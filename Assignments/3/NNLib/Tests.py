import unittest
import numpy as np
import ReLU
from LinearLayer import LinearLayer


class LayerTest(unittest.TestCase):

    def setUp(self):
        self.linear_layer = LinearLayer(10, 3)

    def test_linearLayer_forward(self):
        self.linear_layer.set_weights(np.full((10, 3), 0.5))
        self.linear_layer.set_bias(np.full(3, 2))
        out = self.linear_layer.forward(np.full((2, 10), 4))
        self.assertTrue(np.array_equal(out, np.array([[22, 22, 22], [22, 22, 22]])))
        self.assertEqual((2, 3), out.shape)

    def test_ReLU_forward(self):
        x = np.full((4, 4), 1)
        x[2:] = x[2:] - 2
        out = ReLU.forward(x)
        self.assertTrue(np.array_equal(out, np.array([[1,  1,  1,  1], [1,  1,  1,  1], [0, 0, 0, 0], [0, 0, 0, 0]])))

    def test_ReLU_backwards(self):
        x = np.full((4, 4), 5)
        x[:, 2] = x[:, 2] - 9
        out = ReLU.backwards(x)
        self.assertTrue(np.array_equal(out, np.array([[1, 1, 0, 1], [1, 1, 0, 1], [1, 1, 0, 1], [1, 1, 0, 1]])))

    def test_Sigmoid_forwards(self):
        pass

    def test_Sigmoid_backwards(self):
        pass


if __name__ == "__main__":
    unittest.main()