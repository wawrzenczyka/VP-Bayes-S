import numpy as np
from sklearn import metrics


def calculate_metrics(
    y_proba,
    y,
    s_proba,
    s,
    use_s_rule=False,
    no_s_info_for_prediction=False,
    method=None,
    time=None,
):
    y = np.where(y == 1, 1, 0)
    s = np.where(s == 1, 1, 0)

    if use_s_rule:
        y_pred = np.where(s == 1, 1, np.where(y_proba > (1 + s_proba) / 2, 1, 0))
    else:
        if no_s_info_for_prediction == False:
            y_pred = np.where(s == 1, 1, np.where(y_proba > 0.5, 1, 0))
        else:
            y_pred = np.where(y_proba > 0.5, 1, 0)

    accuracy = metrics.accuracy_score(y, y_pred)
    precision = metrics.precision_score(y, y_pred)
    recall = metrics.recall_score(y, y_pred)
    f1 = metrics.f1_score(y, y_pred)
    u_accuracy = metrics.accuracy_score(y[s == 0], y_pred[s == 0])
    u_precision = metrics.precision_score(y[s == 0], y_pred[s == 0])
    u_recall = metrics.recall_score(y[s == 0], y_pred[s == 0])
    u_f1 = metrics.f1_score(y[s == 0], y_pred[s == 0])

    if use_s_rule:
        auc_probas = np.where(s == 1, 1, y_proba - (s_proba / 2))
        auc_probas_v2 = np.where(
            s == 1,
            1,
            np.where(  # to avoid numerical problems
                np.abs(1 - y_proba) > 1e-10,
                (y_proba - s_proba) / (1 - y_proba),  # pure form
                1,
            ),
        )
    else:
        if no_s_info_for_prediction == False:
            auc_probas = np.where(s == 1, 1, y_proba)
            auc_probas_v2 = np.where(s == 1, 1, y_proba)
        else:
            auc_probas = y_proba
            auc_probas_v2 = y_proba

    auc = metrics.roc_auc_score(y, auc_probas)
    u_auc = metrics.roc_auc_score(y[s == 0], auc_probas[s == 0])
    auc_v2 = metrics.roc_auc_score(y, auc_probas_v2)
    u_auc_v2 = metrics.roc_auc_score(y[s == 0], auc_probas_v2[s == 0])

    metric_values = {
        "Method": method,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 score": f1,
        "AUC": auc,
        "AUC-v2": auc_v2,
        "U-Accuracy": u_accuracy,
        "U-Precision": u_precision,
        "U-Recall": u_recall,
        "U-F1 score": u_f1,
        "U-AUC": u_auc,
        "U-AUC-v2": u_auc_v2,
        "Time": time,
    }
    return metric_values
