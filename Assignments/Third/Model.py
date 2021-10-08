import numpy as np

from Assignments.Third.NNLib.EmbeddingLayer import EmbeddingLayer
from Assignments.Third.NNLib.LinearLayer import LinearLayer
from Assignments.Third.NNLib import ReLU, Softmax


def cross_entropy(y, y_pred):
    """
    Cross entropy loss
    :param y: the true label
    :param y_pred: predicted label
    :return: the cross entropy loss
    """
    return -y * np.log(y_pred) + (1 - y) * np.log(1-y_pred)


def d_LCE(y, y_pred):
    """
    derivative of cross entropy
    :param y: true y value
    :param y_pred: predicted y value
    :return: the derivative of the cross entropy loss
    """
    return y_pred - y



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
        self.z = {}
        self.a = {}
        self.dL_dw = {str(i): None for i in range(3)}
        self.dL_db = {str(i): None for i in range(3)}

    def forward(self, x):
        # creat e with n word embeddings
        # Multiply with hidden layer
        # Pass through ReLU
        # Multiply with output layer
        e = np.array([self.embedding_layer.forward(x[:, i]) for i in range(self.memory_depth)])
        self.z['0'] = np.concatenate(e, axis=1)
        self.a['0'] = np.concatenate(e, axis=1)
        self.z['1'] = self.hidden_layer.forward(self.a['0'])
        self.a['1'] = self.ReLu(self.z['1'])
        self.z['2'] = self.output_layer.forward(self.a['1'])
        self.a['2'] = self.softmax(self.z['2'])
        return self.a['2']

    def test_mode(self):
        self.embedding_layer.test_mode()
        self.hidden_layer.test_mode()
        self.output_layer.test_mode()

    def backprop(self, y_true, y_pred, dloss=d_LCE):
        layer = 2
        delta_curr = dloss(y_true, y_pred)
        self.dL_dw[str(layer)] = delta_curr.reshape(np.prod(delta_curr.shape), 1) * self.a[str(layer-1)]
        self.dL_db[str(layer)] = delta_curr
        layers = [self.embedding_layer, self.hidden_layer]
        for l in range(layer-1, -1, -1):
            curr_l = layers[l]
            delta_prev = delta_curr
            tmp = delta_prev.reshape(np.prod(delta_prev.shape), 1) * curr_l.weights
            delta_curr = np.sum(tmp, axis=0, keepdims=True) * ReLU.backwards(self.z[str(l)])
            self.dL_dw[str(l)] = delta_curr.reshape(np.prod(delta_curr.shape), 1) * self.a[str(l)]
            self.dL_db = delta_curr


