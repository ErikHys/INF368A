import numpy as np


class LinearLayer:

    def __init__(self, input_size, output_size, random_seed=27):
        assert isinstance(input_size, int), "input size should be an int, maybe you need to flatten your input data"
        assert isinstance(output_size, int), "output_size should be an int"

        rand = np.random.default_rng(seed=random_seed)
        self.weights = rand.random((output_size, input_size))
        self.bias = rand.random(output_size)
        self.input_size = input_size
        self.output_size = output_size

    def forward(self, x):
        # assert x.shape[1] == self.weights.shape[0], f"Mismatching dimension between input features {x.shape[1]} and " \
        #                                             f"weights {self.weights.shape[0]} "
        return np.matmul(self.weights, x)

    def get_weights(self):
        return self.weights

    def get_bias(self):
        return self.bias

    def set_weights(self, w):
        assert w.shape == self.weights.shape
        self.weights = w

    def set_bias(self, b):
        assert b.shape == self.bias.shape
        self.bias = b

    def input_output_size(self):
        return self.input_size, self.output_size

    def test_mode(self):
        self.weights = np.ones(self.weights.shape)
        self.bias = np.ones(self.bias.shape)

