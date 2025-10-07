import numpy as np
from abc import ABC, abstractmethod
import copy


class BaseEstimator(ABC):
    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    def get_params(self, deep=True):
        params = {}
        for key, value in vars(self).items():
            if not key.endswith('_'):
                params[key] = value
        return params

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def score(self, X, y):
        y_pred = self.predict(X)
        y = np.asarray(y)
        ss_tot = np.sum((y - np.mean(y))**2)
        ss_res = np.sum((y - y_pred)**2)
        return 1 - (ss_res / ss_tot)


class LinearModel(BaseEstimator):
    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = None

    def predict(self, X):
        X = np.asarray(X)
        if self.coef_.ndim == 1:
            return X @ self.coef_ + self.intercept_
        else:
            return X @ self.coef_.T + self.intercept_

    def _preprocess_data(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        if self.fit_intercept:
            X_offset = np.mean(X, axis=0)
            y_offset = np.mean(y, axis=0) if y.ndim == 1 else np.mean(y, axis=0)
            X = X - X_offset
            y = y - y_offset
        else:
            X_offset = np.zeros(X.shape[1])
            y_offset = 0.0 if y.ndim == 1 else np.zeros(y.shape[1])

        return X, y, X_offset, y_offset

    def _set_intercept(self, X_offset, y_offset):
        if self.fit_intercept:
            if self.coef_.ndim == 1:
                self.intercept_ = y_offset - X_offset @ self.coef_
            else:
                self.intercept_ = y_offset - X_offset @ self.coef_.T
        else:
            self.intercept_ = 0.0


class BaseTransformer(ABC):
    @abstractmethod
    def fit(self, X, y=None):
        pass

    @abstractmethod
    def transform(self, X):
        pass

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def get_params(self, deep=True):
        params = {}
        for key, value in vars(self).items():
            if not key.endswith('_'):
                params[key] = value
        return params

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

def clone(estimator, safe=True):
    estimator_type = type(estimator)

    if hasattr(estimator, 'get_params'):
        params = estimator.get_params(deep=False)

        cloned_params = {}
        for key, value in params.items():
            if hasattr(value, 'get_params') and not isinstance(value, type):
                cloned_value = clone(value, safe=safe)
            elif isinstance(value, list):
                cloned_value = [
                    clone(item, safe=safe) if hasattr(item, 'get_params') else copy.deepcopy(item)
                    for item in value
                ]
            else:
                cloned_value = copy.deepcopy(value)
            cloned_params[key] = cloned_value

        new_estimator = estimator_type(**cloned_params)
    else:
        new_estimator = copy.deepcopy(estimator)

    return new_estimator