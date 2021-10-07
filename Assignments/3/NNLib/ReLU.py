import numpy as np


def forward(x):
    return x * (x > 0)


def backwards(x):
    return np.ones(x.shape) * (x > 0)
