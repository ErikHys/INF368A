import unittest
import numpy as np
from Assignments.Third.NNLib import ReLU, Softmax
from Assignments.Third.NNLib.LinearLayer import LinearLayer
from Assignments.Third.Model import MyFFLM


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

    def test_Softmax_forwards(self):
        # x = np.array([[1, 2, 3, 4, 5, 6],
        #               [0, 1, 0, 0, 0, 0],
        #               [1, 1, 1, 1, 1, 1],
        #               [2, 1, 0, 0, 0, 0]])
        x = np.array([[1, 0], [1000, 0]])
        print(Softmax.forward(x))

    def test_myFFLM(self):
        x = np.array([[[0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 1]],
                      [[0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 1]],
                      [[1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]],
                      [[0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 1]]])
        my_model = MyFFLM(6, 2, 3)
        y = my_model.forward(x)
        print(np.sum(y, axis=1, keepdims=True))


if __name__ == "__main__":
    unittest.main()