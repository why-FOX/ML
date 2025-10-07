import numpy as np
from scipy import linalg, optimize
from .base import LinearModel


class LinearRegression(LinearModel):
    def __init__(self, fit_intercept=True):
        super().__init__(fit_intercept=fit_intercept)

    def fit(self, X, y):
        X, y, X_offset, y_offset = self._preprocess_data(X, y)

        self.coef_, _, self.rank_, self.singular_ = linalg.lstsq(X, y)
        self.coef_ = self.coef_.T

        if y.ndim == 1:
            self.coef_ = np.ravel(self.coef_)

        self._set_intercept(X_offset, y_offset)
        return self


class Ridge(LinearModel):
    def __init__(self, alpha=1.0, fit_intercept=True, solver='auto'):
        super().__init__(fit_intercept=fit_intercept)
        self.alpha = alpha
        self.solver = solver

    def fit(self, X, y):
        X, y, X_offset, y_offset = self._preprocess_data(X, y)

        n_samples, n_features = X.shape

        if self.solver == 'auto':
            if n_samples > n_features:
                solver = 'cholesky'
            else:
                solver = 'svd'
        else:
            solver = self.solver

        if y.ndim == 1:
            y = y.reshape(-1, 1)

        if solver == 'cholesky':
            self.coef_ = self._solve_cholesky(X, y, self.alpha)
        elif solver == 'svd':
            self.coef_ = self._solve_svd(X, y, self.alpha)
        else:
            raise ValueError(f"Unknown solver: {solver}")

        if y.shape[1] == 1:
            self.coef_ = np.ravel(self.coef_)

        self._set_intercept(X_offset, y_offset)
        return self

    def _solve_cholesky(self, X, y, alpha):
        n_features = X.shape[1]
        A = X.T @ X
        Xy = X.T @ y

        A.flat[::n_features + 1] += alpha
        coef = linalg.solve(A, Xy, assume_a='pos')
        return coef.T

    def _solve_svd(self, X, y, alpha):
        U, s, Vt = linalg.svd(X, full_matrices=False)

        idx = s > 1e-15
        s_nnz = s[idx][:, None]
        UTy = U.T @ y

        d = np.zeros((s.shape[0], y.shape[1]))
        d[idx] = s_nnz / (s_nnz**2 + alpha)

        d_UT_y = d * UTy
        return (Vt.T @ d_UT_y).T


class Lasso(LinearModel):
    def __init__(self, alpha=1.0, fit_intercept=True, max_iter=1000, tol=1e-4):
        super().__init__(fit_intercept=fit_intercept)
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X, y):
        X, y, X_offset, y_offset = self._preprocess_data(X, y)

        n_samples, n_features = X.shape
        self.coef_ = np.zeros(n_features)

        X_norms_sq = np.sum(X**2, axis=0)

        for iteration in range(self.max_iter):
            coef_old = self.coef_.copy()

            for j in range(n_features):
                residual = y - X @ self.coef_
                rho = X[:, j] @ residual + X_norms_sq[j] * self.coef_[j]

                if X_norms_sq[j] < 1e-10:
                    self.coef_[j] = 0.0
                else:
                    self.coef_[j] = self._soft_threshold(rho, self.alpha * n_samples) / X_norms_sq[j]

            if np.max(np.abs(self.coef_ - coef_old)) < self.tol:
                break

        self._set_intercept(X_offset, y_offset)
        return self

    def _soft_threshold(self, z, gamma):
        if z > gamma:
            return z - gamma
        elif z < -gamma:
            return z + gamma
        else:
            return 0.0
