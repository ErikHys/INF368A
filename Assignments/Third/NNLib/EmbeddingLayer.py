import numpy as np


class EmbeddingLayer:

    def __init__(self, vocab_size, embedding_size, random_seed=27):
        rand = np.random.default_rng(seed=random_seed)
        self.embedding = rand.random((embedding_size, vocab_size))

    def forward(self, x):
        return np.matmul(self.embedding, x.T).T

    def test_mode(self):
        self.embedding = np.ones(self.embedding.shape)
