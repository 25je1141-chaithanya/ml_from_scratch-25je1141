import numpy as np

class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def fit(self, X, y, epochs=1000):
        for _ in range(epochs):
            out = self.forward(X)
            loss_grad = out - y
            self.backward(loss_grad)
