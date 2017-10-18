import numpy as np


def train_slit(n, split_size=50000):
    idx = np.arange(n)
    np.random.seed(123)
    np.random.shuffle(idx)
    split = np.zeros(n, dtype=np.bool)
    split[idx[:-split_size]] = True
    return split
