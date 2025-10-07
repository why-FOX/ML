import numpy as np
from itertools import combinations_with_replacement
from scipy.interpolate import BSpline
from .base import BaseTransformer


class PolynomialFeatures(BaseTransformer):
    def __init__(self, degree=2, include_bias=True):
        self.degree = degree
        self.include_bias = include_bias
        self.n_input_features_ = None
        self.n_output_features_ = None

    def fit(self, X, y=None):
        X = np.asarray(X)
        n_samples, n_features = X.shape
        self.n_input_features_ = n_features

        combinations = self._combinations(n_features, self.degree, self.include_bias)
        self.n_output_features_ = len(combinations)

        return self

    def transform(self, X):
        X = np.asarray(X)
        n_samples, n_features = X.shape

        combinations = self._combinations(n_features, self.degree, self.include_bias)
        n_output_features = len(combinations)

        XP = np.empty((n_samples, n_output_features), dtype=X.dtype)

        for i, comb in enumerate(combinations):
            XP[:, i] = np.prod(X[:, comb], axis=1)

        return XP

    def _combinations(self, n_features, degree, include_bias):
        start = 0 if include_bias else 1
        return [comb for d in range(start, degree + 1)
                for comb in combinations_with_replacement(range(n_features), d)]


class SplineTransformer(BaseTransformer):
    def __init__(self, n_knots=10, degree=3):
        self.n_knots = n_knots
        self.degree = degree
        self.knots_ = None
        self.n_features_in_ = None

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float64)
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features

        self.knots_ = []
        for j in range(n_features):
            x_min, x_max = X[:, j].min(), X[:, j].max()
            margin = (x_max - x_min) * 0.01
            knots = np.linspace(x_min - margin, x_max + margin, self.n_knots)

            knots_full = np.concatenate([
                np.repeat(knots[0], self.degree),
                knots,
                np.repeat(knots[-1], self.degree)
            ])
            self.knots_.append(knots_full)

        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        n_samples, n_features = X.shape

        n_splines = self.n_knots + self.degree - 1
        output_features = []

        for j in range(n_features):
            knots = self.knots_[j]
            x = X[:, j]

            spline_features = np.zeros((n_samples, n_splines))

            for i in range(n_splines):
                coef = np.zeros(n_splines)
                coef[i] = 1.0

                bspline = BSpline(knots, coef, self.degree, extrapolate=True)
                spline_features[:, i] = bspline(x)

            output_features.append(spline_features)

        return np.hstack(output_features)
