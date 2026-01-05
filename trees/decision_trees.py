import numpy as np

class DecisionTree:
    def __init__(self, max_depth=5):
        self.max_depth = max_depth

    def _gini(self, y):
        classes, counts = np.unique(y, return_counts=True)
        prob = counts / counts.sum()
        return 1 - np.sum(prob**2)

    def fit(self, X, y, depth=0):
        if depth == self.max_depth or len(np.unique(y)) == 1:
            return np.bincount(y).argmax()

        best_feat = 0
        best_thresh = 0
        best_gini = float("inf")

        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            for t in thresholds:
                left = y[X[:, feature] <= t]
                right = y[X[:, feature] > t]
                g = self._gini(left) + self._gini(right)
                if g < best_gini:
                    best_gini = g
                    best_feat = feature
                    best_thresh = t

        return {
            "feature": best_feat,
            "threshold": best_thresh
        }
