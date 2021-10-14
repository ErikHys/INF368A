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

    def __init__(self, vocab_size, embedding_size, learning_rate=0.01, memory_depth=3):
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
        self.embedding_size = embedding_size
        self.vocab_size = vocab_size
        self.learning_rate = learning_rate
        self.z = {}
        self.a = {}
        self.dL_dw = {str(i): None for i in range(3)}
        self.dL_db = {str(i): None for i in range(3)}
        self.df = {'0': lambda x: x, '1': ReLU.backwards, '2': lambda x: x}
        self.dL_dw['0'] = [None for _ in range(memory_depth)]

    def forward(self, x):
        # creat e with n word embeddings
        # Multiply with hidden layer
        # Pass through ReLU
        # Multiply with output layer
        x = x.T
        self.a['-1'] = x
        e = np.array([self.embedding_layer.forward(x[:, i]) for i in range(self.memory_depth)])
        self.z['0'] = np.concatenate(e, axis=0).reshape((self.memory_depth*self.embedding_size, 1))
        self.a['0'] = self.z['0']
        self.z['1'] = self.hidden_layer.forward(self.a['0'])
        self.a['1'] = self.ReLu(self.z['1'])
        self.z['2'] = self.output_layer.forward(self.a['1'])
        self.a['2'] = self.softmax(self.z['2'])
        return self.a['2'].reshape(1, self.vocab_size)

    def test_mode(self):
        self.embedding_layer.test_mode()
        self.hidden_layer.test_mode()
        self.output_layer.test_mode()

    def backprop(self, y_true, y_pred, dloss=d_LCE):
        dl_da2 = dloss(y_true, y_pred)
        dl_dz2 = np.matmul(dl_da2.T, self.a['1'].T)
        dl_da1 = np.matmul(dl_dz2, self.df['1'](self.z['1']))
        dl_dz1 = dl_da1 * self.a['0']
        dl_da0 = dl_dz1 * self.df['0'](self.z['0'])
        dl_dz0 = dl_da0 * self.a['-1']
        self.dL_dw['2'] = dl_dz2
        self.dL_dw['1'] = dl_dz1
        self.dL_dw['0'] = dl_dz0
        self.dL_db['2'] = dl_da2
        self.dL_db['1'] = dl_da1
        self.dL_db['0'] = dl_da0
        self.update()

    def update(self):
        for i in range(self.memory_depth):
            self.embedding_layer.weights -= self.learning_rate*self.dL_dw['0'][i]
        self.hidden_layer.weights -= self.learning_rate*self.dL_dw['1']
        self.hidden_layer.bias -= self.learning_rate*self.dL_db['1']
        self.output_layer.weights -= self.learning_rate * self.dL_dw['2']
        self.output_layer.bias -= self.learning_rate * self.dL_db['2']
