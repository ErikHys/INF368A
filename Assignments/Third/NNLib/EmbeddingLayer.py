import numpy as np


class EmbeddingLayer:

    def __init__(self, vocab_size, embedding_size, random_seed=27):
        rand = np.random.default_rng(seed=random_seed)
        self.weights = rand.random((embedding_size, vocab_size))

    def forward(self, x):
        return np.matmul(self.weights, x)

    def test_mode(self):
        self.weights = np.ones(self.weights.shape)
