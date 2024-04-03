# %%
import numpy as np
from sklearn import metrics
from sklearn.ensemble import IsolationForest
from sklearn.svm import SVC


def train_2_step(train_samples, config, num_exp):
    print(f"Exp {num_exp} ||| Two-step PU training")

    x_train, _, s_train = train_samples
    x_l = x_train[s_train == 1]
    x_u = x_train[s_train == 0]

    pi_pl, pi_u, pi_pu = (
        config["pi_pl"],
        config["pi_u"],
        config["pi_pu"],
    )
    pu_to_u_ratio = pi_pu / pi_u

    contamination = 0.001
    contamination = min(max(contamination, 0.004), 0.1)
    occ = IsolationForest(random_state=num_exp, contamination=contamination)
    occ.fit(x_u)

    cleanup_samples = int(pu_to_u_ratio * len(x_u))
    scores = occ.score_samples(x_u)
    reliable_negatives_idx = np.argsort(scores)[cleanup_samples:]

    x_n = x_u[reliable_negatives_idx]

    X = np.concatenate([x_l, x_n])
    y = np.concatenate([np.ones(len(x_l)), np.zeros(len(x_n))])

    clf = SVC(gamma=0.001)
    clf.fit(X, y)

    return clf


def eval_2_step(clf, test_samples):
    x_test, y_test, _ = test_samples
    y_pred = clf.predict(x_test)

    y_test = np.where(y_test == 1, 1, 0)
    print(metrics.classification_report(y_test, y_pred))
    accuracy = metrics.accuracy_score(y_test, y_pred)
    precision = metrics.precision_score(y_test, y_pred)
    recall = metrics.recall_score(y_test, y_pred)
    f1 = metrics.f1_score(y_test, y_pred)

    return accuracy, precision, recall, f1
