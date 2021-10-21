import numpy as np


class EmbeddingLayer:

    def __init__(self, vocab_size, embedding_size, random_seed=27):
        rand = np.random.default_rng(seed=random_seed)
        self.weights = rand.random((embedding_size, vocab_size), dtype='float32')

    def forward(self, x):
        assert x.shape[0] == self.weights.shape[1], f"Mismatching dimension between input features {x.shape[1]} and " \
                                                    f"weights {self.weights.shape[0]}"
        return np.matmul(self.weights, x.T)

    def test_mode(self):
        self.weights = np.ones(self.weights.shape)
