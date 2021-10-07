import numpy as np

from Assignments.Third.NNLib.EmbeddingLayer import EmbeddingLayer
from Assignments.Third.NNLib.LinearLayer import LinearLayer
from Assignments.Third.NNLib import ReLU, Softmax


class MyFFLM:

    def __init__(self, vocab_size, embedding_size, memory_depth=3):
        """

        :param vocab_size:
        :param embedding_size:
        :param memory_depth:
        """
        self.embedding_layer = EmbeddingLayer(vocab_size, embedding_size)
        self.hidden_layer = LinearLayer(memory_depth*embedding_size, embedding_size)
        self.ReLu = ReLU.forward
        self.output_layer = LinearLayer(embedding_size, vocab_size)
        self.softmax = Softmax.forward
        self.memory_depth = memory_depth

    def forward(self, x):
        # creat e with n word embeddings
        # Multiply with hidden layer
        # Pass through ReLU
        # Multiply with output layer
        o = x[:, 0]
        p = x[:, 1]
        t = self.embedding_layer.forward(x[0, 0])
        u = self.embedding_layer.forward(x[:, 0])
        e = np.array([self.embedding_layer.forward(x[:, i]) for i in range(self.memory_depth)])
        e = np.concatenate(e, axis=1)
        z1 = self.hidden_layer.forward(e)
        a1 = self.ReLu(z1)
        z2 = self.output_layer.forward(a1)
        out = self.softmax(z2)
        return out
