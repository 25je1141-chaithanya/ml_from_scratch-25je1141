import numpy as np
from linear_regression import LinearRegression

class PolynomialRegression:
    def __init__(self, degree=2):
        self.degree = degree
        self.model = LinearRegression()

    def _poly_features(self, X):
        features = [X**i for i in range(1, self.degree + 1)]
        return np.concatenate(features, axis=1)

    def fit(self, X, y):
        X_poly = self._poly_features(X)
        self.model.fit(X_poly, y)

    def predict(self, X):
        X_poly = self._poly_features(X)
        return self.model.predict(X_poly)
