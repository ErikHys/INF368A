import numpy as np


def forward(x):
    return x * (x > 0)


def backwards(x):
    return (x > 0).astype('float32')
