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
    epsilon = 1e-12
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    N = y.shape[0]
    ce = -np.dot(y, np.log(y_pred.flatten()))
    return ce


def d_LCE(y, y_pred):
    """
    derivative of cross entropy
    :param y: true y value
    :param y_pred: predicted y value
    :return: the derivative of the cross entropy loss
    """
    return y_pred - y


def scale(X):
    """
    :param X: Input data
    :return: A scaled input data where absolute value of X.max() is 1
    """
    if not np.isfinite(X).all():
        return X * 0
    denom = np.abs(X).max()
    denom = denom + (denom == 0)
    return X / denom


class MyFFLM:

    def __init__(self, vocab_size, embedding_size, learning_rate=0.01, memory_depth=3):
        """
        :param vocab_size:
        :param embedding_size:
        :param memory_depth:
        """
        self.embedding_layer = EmbeddingLayer(vocab_size, embedding_size)
        self.hidden_layer = LinearLayer(memory_depth*embedding_size, 512)
        self.ReLu = ReLU.forward
        self.output_layer = LinearLayer(512, vocab_size)
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
        """create e with n word embeddings
        Multiply with hidden layer
        Pass through ReLU
        Multiply with output layer
        :param Input data, a single case
        :return probability distribution of predicted words
        """
        self.a['-1'] = x
        e = np.array([self.embedding_layer.forward(x[i]) for i in range(self.memory_depth)])
        self.z['0'] = np.concatenate(e, axis=0)

        self.a['0'] = self.z['0']
        self.z['1'] = self.hidden_layer.forward(self.a['0'])

        self.a['1'] = self.ReLu(self.z['1'])
        self.z['2'] = self.output_layer.forward(self.a['1'])

        self.a['2'] = self.softmax(self.z['2']).reshape(1, self.vocab_size)
        return self.a['2']

    def test_mode(self):
        """
        Activates test mode for each layer where all weights = 1
        """
        self.embedding_layer.test_mode()
        self.hidden_layer.test_mode()
        self.output_layer.test_mode()

    def backprop(self, y_true, dloss=d_LCE):
        """
        Calculates gradients for each layer, then updates the weights
        :param y_true: one hot encoded y label
        :param dloss: Derived loss function
        """

        # Calculate the last layer
        delta = dloss(y_true, self.a['2']) * self.a['2']
        dl_da2 = delta
        # Multiply with input to last layer to get weight gradient
        dl_dz2 = np.matmul(delta.T, self.a['1'].reshape(1, -1))
        # Calculate the hidden layer gradient
        tmp = np.matmul(delta, self.output_layer.weights)
        s = np.sum(tmp, axis=1,  keepdims=True)
        df_z1 = self.df['1'](self.z['1'])
        delta = np.matmul(s.reshape(-1, 1), df_z1.reshape(1, -1))
        dl_da1 = delta
        dl_dz1 = np.matmul(dl_da1.reshape(-1, 1), self.a['0'].reshape(1, -1))
        # Repeat for final layer
        tmp = np.matmul(delta, self.hidden_layer.weights)
        s = np.sum(tmp, axis=1,  keepdims=True)
        df_z0 = self.df['0'](self.z['0'])
        delta = np.matmul(s.reshape(-1, 1), df_z0.reshape(1, -1))
        q = int((delta.shape[1]/self.memory_depth))
        # Split gradients into each word so dimensions fit with embedding layer
        d = np.array([delta[:, i: i+q] for i in range(self.memory_depth)]).reshape(self.memory_depth, self.embedding_size)
        dl_da0 = d
        dl_dz0 = np.array([np.matmul(dl_da0[i].reshape(-1, 1), self.a['-1'][i].reshape(1, -1)) for i in range(self.memory_depth)])

        # Store gradients in dictionaries
        self.dL_dw['2'] = scale(dl_dz2)
        self.dL_dw['1'] = scale(dl_dz1)
        for i in range(self.memory_depth):
            self.dL_dw['0'][i] = scale(dl_dz0[i])
        self.dL_db['2'] = scale(dl_da2)
        self.dL_db['1'] = scale(dl_da1)
        self.dL_db['0'] = scale(dl_da0)
        self._update()

    def _update(self):
        """
        Preform gradient descent
        """
        for i in range(self.memory_depth):
            self.embedding_layer.weights -= self.learning_rate*self.dL_dw['0'][i]
        self.hidden_layer.weights -= self.learning_rate*self.dL_dw['1']
        self.hidden_layer.bias -= self.learning_rate*self.dL_db['1'].flatten()
        self.output_layer.weights -= self.learning_rate * self.dL_dw['2']
        self.output_layer.bias -= self.learning_rate * self.dL_db['2'].flatten()
