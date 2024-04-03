# %%
import numpy as np
import statsmodels.api as sm
from sklearn import metrics
from sklearn.ensemble import IsolationForest

from external.logistic_regression import LogisticRegression


def train_em_pu_cc(train_samples, config, num_exp):
    print(f"Exp {num_exp} ||| EM-PU training")

    x_train, _, s_train = train_samples
    x_l = x_train[s_train == 1]
    x_u = x_train[s_train == 0]

    pi_pl, pi_u, pi_pu = (
        config["pi_pl"],
        config["pi_u"],
        config["pi_pu"],
    )
    pi = pi_pl + pi_pu

    y_hat = np.where(s_train == 1, 1, pi)
    prev_y_hat = y_hat

    i = 0
    while True:
        clf = LogisticRegression()
        clf.fit(x_train, y_hat)

        y_pred_before = clf.predict(x_train)
        clf.intercept_ -= np.log((len(x_l) + pi * len(x_u)) / (pi * len(x_u)))
        y_pred_after = clf.predict(x_train)

        y_hat = np.where(s_train == 1, 1, clf.predict_proba(x_train))

        i += 1
        predictions_change = np.mean(np.abs(y_hat - prev_y_hat)[s_train == 0])
        print(f"Iter {i} ||| change: {predictions_change:.5f}")
        if i == 20 or predictions_change < 1e-3:
            break
        prev_y_hat = y_hat

    return clf


def eval_em_pu_cc(clf, test_samples):
    x_test, y_test, _ = test_samples
    y_pred = clf.predict(x_test)

    y_test = np.where(y_test == 1, 1, 0)
    print(metrics.classification_report(y_test, y_pred))
    accuracy = metrics.accuracy_score(y_test, y_pred)
    precision = metrics.precision_score(y_test, y_pred)
    recall = metrics.recall_score(y_test, y_pred)
    f1 = metrics.f1_score(y_test, y_pred)

    return accuracy, precision, recall, f1
