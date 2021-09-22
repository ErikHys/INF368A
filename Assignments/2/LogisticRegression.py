import math
import random


def vector_mult(x, w, b):
    return sum(x[i] * w[i] for i in range(len(x))) + b


def cross_entropy(y, y_pred):
    return -y*math.log(y_pred) + (1 - y) * math.log(1-y_pred)


def d_LCE(y, y_pred):
    return y_pred - y


class LogisticRegression:

    def __init__(self, feature_length):
        self.feature_length = feature_length
        self.weights = [random.random() for _ in range(self.feature_length)]
        self.bias = random.random()

    def predict(self, x):
        return 1 / (1 + math.exp(-vector_mult(x, self.weights, self.bias)))

    def gradient(self, x, y, d_loss_func):
        y_pred = self.predict(x)
        d_loss = d_loss_func(y, y_pred)
        gradients = [x[i]*d_loss for i in range(self.feature_length)]
        return gradients, d_loss

    def gradient_descent(self, gradients, learning_rate):
        new_w = [self.weights[i] - learning_rate * gradients[0][i] for i in range(self.feature_length)]
        new_b = self.bias - learning_rate * gradients[1]
        return new_w, new_b

    def train(self, data, labels, learning_rate=0.1, d_loss_func=d_LCE):
        for x, y in zip(data, labels):
            gradients = self.gradient(x, y, d_loss_func)
            new_w, new_b = self.gradient_descent(gradients, learning_rate)
            self.weights = new_w
            self.bias = new_b

    def set_weights(self, w, b):
        self.weights = w
        self.bias = b


def run_example():
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


# If you wish to try many updates to see a bigger change in the predicted value run then next part
# it does a 100 updates to the weights
def run_example2():
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


