import math
import random


def vector_dot(x, w, b):
    """
    Simple dot product of vector x and w plus bias
    :param x: first vector
    :param w: second vector
    :param b: bias
    :return: XW + b
    """
    return sum(x[i] * w[i] for i in range(len(x))) + b


def cross_entropy(y, y_pred):
    """
    Cross entropy loss
    :param y: the true label
    :param y_pred: predicted label
    :return: the cross entropy loss
    """
    return -y*math.log(y_pred) + (1 - y) * math.log(1-y_pred)


def d_LCE(y, y_pred):
    """
    derivative of cross entropy
    :param y: true y value
    :param y_pred: predicted y value
    :return: the derivative of the cross entropy loss
    """
    return y_pred - y


class LogisticRegression:

    def __init__(self, feature_length):
        """
        Store length of a single data point and randomly initialize weights and bias
        :param feature_length:
        """
        self.feature_length = feature_length
        self.weights = [random.random() for _ in range(self.feature_length)]
        self.bias = random.random()

    def predict(self, x):
        """
        The sigmoid of the dot product of input and weights plus bias.
        :param x: data to do prediction on
        :return: predicted y value
        """
        return 1 / (1 + math.exp(-vector_dot(x, self.weights, self.bias)))

    def gradient(self, x, y, d_loss_func):
        """
        Calculates the gradient based on the loss
        :param x: input features
        :param y: true y values
        :param d_loss_func: a derivative of the loss function to use
        :return: a tuple with the gradients of the weights and bias
        """
        y_pred = self.predict(x)
        d_loss = d_loss_func(y, y_pred)
        gradients = [x[i]*d_loss for i in range(self.feature_length)]
        return gradients, d_loss

    def gradient_descent(self, gradients, learning_rate):
        """
        Does one update of gradient descent e.g. w_i = w_i - learning_rate * gradient
        :param gradients: a vector of gradients corresponding to the weights
        :param learning_rate: how big of a step the gradient descent should do
        :return: a tuple with a new weight vector and a new bias
        """
        new_w = [self.weights[i] - learning_rate * gradients[0][i] for i in range(self.feature_length)]
        new_b = self.bias - learning_rate * gradients[1]
        return new_w, new_b

    def train(self, data, labels, learning_rate=0.1, d_loss_func=d_LCE):
        """
        Trains the weights on new data
        :param data: set of features
        :param labels: set of labels
        :param learning_rate: how big of a step the gradient descent should do
        :param d_loss_func: a derivative of a loss function
        """
        for x, y in zip(data, labels):
            gradients = self.gradient(x, y, d_loss_func)
            new_w, new_b = self.gradient_descent(gradients, learning_rate)
            self.weights = new_w
            self.bias = new_b

    def set_weights(self, w, b):
        """
        A function to set weights and bias to specific values
        :param w: new weight vector
        :param b: new bias
        """
        self.weights = w
        self.bias = b


def run_example():
    """
    Runs the example section 5.4.3 page 87 from the book
    Speech and Language Processing
    by Dan Jurafsky and James H. Martin
    prints the results to console
    """
    x = [3, 2]
    w = [0, 0]
    b = 0
    y = 1
    alpha = 0.1

    log_reg = LogisticRegression(len(x))
    log_reg.set_weights(w, b)
    print("Prediction before update:", log_reg.predict(x))
    log_reg.train([x], [y], alpha)
    print("Prediction after update:", log_reg.predict(x))
    print("Weights:", log_reg.weights, "Bias:", log_reg.bias)


def run_example2():
    # If you wish to try many updates to see a bigger change in the predicted value run then next part
    # it does a 100 updates to the weights
    x = [3, 2]
    y = 1
    alpha = 0.1
    log_reg = LogisticRegression(len(x))
    for i in range(100):
        print(log_reg.predict(x))
        log_reg.train([x], [y], alpha)
        print(log_reg.predict(x))
        if (i + 1) % 10 == 0:
            print("Weights:", log_reg.weights, "Bias:", log_reg.bias)


