import numpy as np


class Pipeline:
    def __init__(self, steps):
        self.steps = steps

    def fit(self, X, y):
        X_transformed = X
        for i, (name, transformer) in enumerate(self.steps[:-1]):
            X_transformed = transformer.fit_transform(X_transformed, y)

        final_estimator = self.steps[-1][1]
        final_estimator.fit(X_transformed, y)

        return self

    def predict(self, X):
        X_transformed = X
        for i, (name, transformer) in enumerate(self.steps[:-1]):
            X_transformed = transformer.transform(X_transformed)

        final_estimator = self.steps[-1][1]
        return final_estimator.predict(X_transformed)

    def score(self, X, y):
        y_pred = self.predict(X)
        y = np.asarray(y)
        ss_tot = np.sum((y - np.mean(y))**2)
        ss_res = np.sum((y - y_pred)**2)
        return 1 - (ss_res / ss_tot)

    def get_params(self, deep=True):
        params = {'steps': self.steps}
        if deep:
            for name, estimator in self.steps:
                if hasattr(estimator, 'get_params'):
                    for key, value in estimator.get_params(deep=True).items():
                        params[f'{name}__{key}'] = value
                else:
                    for key, value in vars(estimator).items():
                        if not key.endswith('_'):
                            params[f'{name}__{key}'] = value
        return params

    def set_params(self, **params):
        if 'steps' in params:
            self.steps = params.pop('steps')

        for key, value in params.items():
            if '__' in key:
                step_name, param_name = key.split('__', 1)
                for name, estimator in self.steps:
                    if name == step_name:
                        if hasattr(estimator, 'set_params'):
                            estimator.set_params(**{param_name: value})
                        else:
                            setattr(estimator, param_name, value)
                        break
        return self

    def __getitem__(self, index):
        if isinstance(index, int):
            return self.steps[index][1]
        elif isinstance(index, str):
            for name, estimator in self.steps:
                if name == index:
                    return estimator
        raise KeyError(f"Step {index} not found")


def make_pipeline(*steps):
    names = [f"step_{i}" for i in range(len(steps))]
    return Pipeline(list(zip(names, steps)))
