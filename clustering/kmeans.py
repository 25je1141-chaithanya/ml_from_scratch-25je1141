import numpy as np

class KMeans:
    def __init__(self, k=3, epochs=100):
        self.k = k
        self.epochs = epochs

    def fit(self, X):
        self.centroids = X[np.random.choice(len(X), self.k, replace=False)]

        for _ in range(self.epochs):
            clusters = [[] for _ in range(self.k)]

            for x in X:
                idx = np.argmin(np.linalg.norm(x - self.centroids, axis=1))
                clusters[idx].append(x)

            self.centroids = np.array([
                np.mean(cluster, axis=0) for cluster in clusters
            ])

    def predict(self, X):
        return np.array([
            np.argmin(np.linalg.norm(x - self.centroids, axis=1))
            for x in X
        ])
