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
        self.a['-1'] = x
        e = np.array([self.embedding_layer.forward(x[:, i]) for i in range(self.memory_depth)])
        self.z['0'] = np.concatenate(e, axis=0)
        self.a['0'] = np.concatenate(e, axis=0)
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
        deltas = {str(i): None for i in range(3)}
        last_layer = None
        layers = [self.embedding_layer, self.hidden_layer, self.output_layer]
        for n, (yy_t, yy_p) in enumerate(zip(y_true, y_pred)):
            for i_layer in range(2, -1, -1):
                layer = layers[i_layer]
                if i_layer == 2:
                    e_i = dloss(yy_t, yy_p)
                else:
                    e_i = np.zeros(layer.weights.shape)
                    for i in range(layer.weights.shape[0]):
                        e_i[i] = 0
                        for k in range(last_layer.weights.shape[0]):
                            weights_layer_plus1_k_i = last_layer.weights[k, i]
                            last_delta = deltas[str(i_layer + 1)][k]
                            e_i[i] += last_delta + weights_layer_plus1_k_i
                f_prime = self.df[str(i_layer)](self.z[str(i_layer)])
                if i_layer > 0:
                    delta = np.matmul(e_i.T, f_prime)
                    deltas[str(i_layer)] = delta
                    self.dL_db[str(i_layer)] = delta
                    new_dL_dw = np.zeros(layer.weights.shape)
                    for i in range(new_dL_dw.shape[0]):
                        for j in range(new_dL_dw.shape[1]):
                            new_dL_dw[i, j] = delta[i, 0] * self.a[str(i_layer-1)][j, 0]
                    self.dL_dw[str(i_layer)] = new_dL_dw
                    last_layer = layer
                else:
                    for m, k in enumerate(range(0, f_prime.shape[0], 2)):
                        delta = np.matmul(e_i.T, f_prime[k:k+2])
                        deltas[str(i_layer)] = delta
                        self.dL_db[str(i_layer)] = delta
                        new_dL_dw = np.zeros(layer.weights.shape)
                        for i in range(new_dL_dw.shape[0]):
                            for j in range(new_dL_dw.shape[1]):
                                new_dL_dw[i, j] = delta[i, 0] * self.a[str(i_layer - 1)][n, m, j]
                        self.dL_dw[str(i_layer)][m] = new_dL_dw
                    last_layer = layer
            self.update()

    def update(self):
        for i in range(self.memory_depth):
            self.embedding_layer.weights -= self.learning_rate*self.dL_dw['0'][i]
        self.hidden_layer.weights -= self.learning_rate*self.dL_dw['1']
        self.hidden_layer.bias -= self.learning_rate*self.dL_db['1']
        self.output_layer.weights -= self.learning_rate * self.dL_dw['2']
        self.output_layer.bias -= self.learning_rate * self.dL_db['2']
