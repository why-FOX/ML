import numpy as np
from .base import BaseTransformer


class StandardScaler(BaseTransformer):
    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float64)
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)
        self.scale_[self.scale_ < 1e-10] = 1.0
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        return (X - self.mean_) / self.scale_
