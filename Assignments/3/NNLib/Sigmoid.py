import numpy as np


def forward(x):
    return 1 / (1 + np.exp(-x))


def backwards(x):
    return forward(x) * (1 - forward(x))
