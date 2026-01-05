import numpy as np

class KNN:
    def __init__(self, k=5):
        self.k = k

    def fit(self, X, y):
        self.X = X
        self.y = y

    def predict(self, X_test):
        preds = []
        for x in X_test:
            dist = np.linalg.norm(self.X - x, axis=1)
            k_idx = np.argsort(dist)[:self.k]
            labels = self.y[k_idx]
            preds.append(np.bincount(labels).argmax())
        return np.array(preds)
