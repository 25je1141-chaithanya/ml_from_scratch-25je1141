import numpy as np

class ReLU:
    def forward(self, Z):
        self.Z = Z
        return np.maximum(0, Z)

    def backward(self, dA):
        return dA * (self.Z > 0)


class Sigmoid:
    def forward(self, Z):
        self.A = 1 / (1 + np.exp(-Z))
        return self.A

    def backward(self, dA):
        return dA * self.A * (1 - self.A)


class Softmax:
    def forward(self, Z):
        exp = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        self.A = exp / np.sum(exp, axis=1, keepdims=True)
        return self.A

    def backward(self, y_true):
        return self.A - y_true
