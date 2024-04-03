import numpy as np
import scipy.optimize


def add_bias(X: np.array) -> np.array:
    return np.insert(X, 0, 1, axis=1)


def sigma(s):
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        result = np.where(s > 0, 1 / (1 + np.exp(-s)), np.exp(s) / (np.exp(s) + 1))

        return result


# from sklearn.metrics import log_loss
def my_log_loss(y_true, y_pred, *, eps=1e-15, sample_weight=None):
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    if y_true.shape[1] == 1:
        y_true = np.append(1 - y_true, y_true, axis=1)

    y_pred = np.clip(y_pred, eps, 1 - eps)

    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)
    if y_pred.shape[1] == 1:
        y_pred = np.append(1 - y_pred, y_pred, axis=1)

    y_pred /= y_pred.sum(axis=1)[:, np.newaxis]
    loss = -(y_true * np.log(y_pred)).sum(axis=1)

    return np.average(loss, weights=sample_weight)


def oracle_risk(b, X, y):
    probability = sigma(X @ b)
    return my_log_loss(y, probability)


def oracle_risk_derivative(b, X, y):
    n = X.shape[0]
    sig = sigma(X @ b)

    multiplier = y - sig
    partial_res = np.sum(X * multiplier.reshape(-1, 1), axis=0)

    return -partial_res / n


class LogisticRegression:
    include_bias: bool

    def __init__(self, include_bias: bool = True):
        self.include_bias = include_bias

    def fit(self, X, y):
        if self.include_bias:
            X = add_bias(X)

        b_init = np.random.random(X.shape[1]) / 100
        # b_init = np.zeros(X.shape[1])

        res = scipy.optimize.minimize(
            fun=oracle_risk,
            x0=b_init,
            method="BFGS",
            args=(X, y),
            jac=oracle_risk_derivative,
        )
        self.intercept_ = res.x[0]
        self.coef_ = res.x[1:]

        return self

    def predict_proba(self, X):
        y_proba = sigma(np.matmul(X, self.coef_) + self.intercept_)
        return y_proba

    def predict(self, X):
        y_proba = self.predict_proba(X)
        y_pred = np.where(y_proba > 0.5, 1, 0)
        return y_pred
