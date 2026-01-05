import numpy as np

def train_test_split(X, y, test_size=0.2, shuffle=True):
    if shuffle:
        idx = np.random.permutation(len(X))
        X, y = X[idx], y[idx]

    split = int(len(X) * (1 - test_size))
    return X[:split], X[split:], y[:split], y[split:]


def standardize(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0) + 1e-8
    return (X - mean) / std
