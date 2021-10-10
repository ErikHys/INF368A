import numpy as np


def forward(x):
    max = np.max(x, axis=0, keepdims=True)
    e_x = np.exp(x - max)
    sum = np.sum(e_x, axis=0, keepdims=True)
    f_x = e_x / sum
    return f_x


def backwards(x):
    pass