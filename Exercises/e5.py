import math

l = len("I have heard a lot of great things about Malcolm Gladwell’s writing! Friends and co-workers tell me that his "
        "subjects are interesting and his writing style is easy to follow without talking down to the reader. I was "
        "not disappointed with Outliers.".split())

x_example = [3, 2, 1, 3, 0, 4.19]
w_example = [2.5, -5, -1.2, 0.5, 2.0, 0.7]
b_example = 0.1
x = [3, 1, 1, 3, 1, math.log(l)]
w = [2.3, -4, -1.3, 1, 1.5, 0.8]
b = 0.2


def cross_entropy(y, y_pred):
    return -y*math.log(y_pred) + (1 - y) * math.log(1-y_pred)


def vector_mult(x, w, b):
    return sum(x[i] * w[i] for i in range(len(x))) + b


def sigmoid(x, w, b):
    return 1 / (1 + math.exp(-vector_mult(x, w, b)))


def sigmoid_b(z):
    return 1 / (1 + math.exp(-z))


def d_LCE(y, y_pred):
    return y_pred - y


def gradient(y, x, w, b, d_loss_func):
    y_pred = sigmoid(x, w, b)
    loss = d_loss_func(y, y_pred)
    gradients = [x[i]*loss for i in range(len(w))]
    return gradients, loss


def gradient_descent(gradients, w, b, learning_rate):
    new_w = [w[i] - learning_rate*gradients[0][i] for i in range(len(w))]
    new_b = b - learning_rate*gradients[1]
    return new_w, new_b


print("Exercise 1: Sentiment prediction", sigmoid(x, w, b))
print("Exercise 2:")
print("Loss if y = 1|", cross_entropy(1, sigmoid(x, w, b)))
print("Loss if y = 0|", cross_entropy(0, sigmoid(x, w, b)))
print("Example:")
y = 1
x = [3, 2]
w = [0, 0]
b = 0
alpha = 0.1
print("Example gradient: ", gradient(y, x, w, b, d_LCE))
print("Example weights after gradient descent: ", gradient_descent(gradient(y, x, w, b, d_LCE), w, b, alpha))
print("Exercise Third:")
y = 1
x = [2, 4]
w = [0.5, 0.5]
b = 0.5
alpha = 0.08
print("Exercise gradient: ", gradient(y, x, w, b, d_LCE))
print("Exercise weights after gradient descent: ", gradient_descent(gradient(y, x, w, b, d_LCE), w, b, alpha))
