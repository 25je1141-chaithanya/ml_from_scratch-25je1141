import numpy as np

class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def step(self, layers):
        for layer in layers:
            if hasattr(layer, "W"):
                layer.W -= self.lr * layer.dW
                layer.b -= self.lr * layer.db


class Momentum:
    def __init__(self, lr=0.01, beta=0.9):
        self.lr = lr
        self.beta = beta
        self.v = {}

    def step(self, layers):
        for i, layer in enumerate(layers):
            if hasattr(layer, "W"):
                if i not in self.v:
                    self.v[i] = {"dW": 0, "db": 0}

                self.v[i]["dW"] = self.beta * self.v[i]["dW"] + (1 - self.beta) * layer.dW
                self.v[i]["db"] = self.beta * self.v[i]["db"] + (1 - self.beta) * layer.db

                layer.W -= self.lr * self.v[i]["dW"]
                layer.b -= self.lr * self.v[i]["db"]


class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = {}
        self.v = {}
        self.t = 0

    def step(self, layers):
        self.t += 1

        for i, layer in enumerate(layers):
            if hasattr(layer, "W"):
                if i not in self.m:
                    self.m[i] = {"dW": 0, "db": 0}
                    self.v[i] = {"dW": 0, "db": 0}

                self.m[i]["dW"] = self.beta1 * self.m[i]["dW"] + (1 - self.beta1) * layer.dW
                self.m[i]["db"] = self.beta1 * self.m[i]["db"] + (1 - self.beta1) * layer.db

                self.v[i]["dW"] = self.beta2 * self.v[i]["dW"] + (1 - self.beta2) * (layer.dW ** 2)
                self.v[i]["db"] = self.beta2 * self.v[i]["db"] + (1 - self.beta2) * (layer.db ** 2)

                m_dw = self.m[i]["dW"] / (1 - self.beta1 ** self.t)
                m_db = self.m[i]["db"] / (1 - self.beta1 ** self.t)

                v_dw = self.v[i]["dW"] / (1 - self.beta2 ** self.t)
                v_db = self.v[i]["db"] / (1 - self.beta2 ** self.t)

                layer.W -= self.lr * m_dw / (np.sqrt(v_dw) + self.eps)
                layer.b -= self.lr * m_db / (np.sqrt(v_db) + self.eps)
