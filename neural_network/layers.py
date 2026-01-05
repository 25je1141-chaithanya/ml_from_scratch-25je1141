import numpy as np

class Dense:
    def __init__(self, input_dim, output_dim, init="xavier"):
        if init == "xavier":
            self.W = np.random.randn(input_dim, output_dim) * np.sqrt(1 / input_dim)
        elif init == "he":
            self.W = np.random.randn(input_dim, output_dim) * np.sqrt(2 / input_dim)
        else:
            self.W = np.random.randn(input_dim, output_dim) * 0.01

        self.b = np.zeros((1, output_dim))

    def forward(self, X):
        self.X = X
        return np.dot(X, self.W) + self.b

    def backward(self, dZ):
        self.dW = np.dot(self.X.T, dZ)
        self.db = np.sum(dZ, axis=0, keepdims=True)
        return np.dot(dZ, self.W.T)
