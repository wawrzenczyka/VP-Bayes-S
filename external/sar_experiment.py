import copy
import json
import os
import threading
import time

import numpy as np
import tensorflow as tf
import torch
from sarpu.experiments import evaluate_all
from sarpu.pu_learning import pu_learn_sar_em
from sklearn import metrics as sk_metrics


class BaseThreadedExperiment(threading.Thread):
    def __init__(
        self, train_samples, test_samples, experiment_id, c, config, method_dir, sem
    ):
        threading.Thread.__init__(self)
        self.train_samples, self.test_samples = train_samples, test_samples
        self.experiment_id = experiment_id
        self.c = c
        self.config = copy.deepcopy(config)
        self.method_dir = method_dir
        self.sem = sem

    def run(self):
        self.sem.acquire()
        self.train_and_test(self.train_samples, self.test_samples)
        self.sem.release()

    def train_and_test(self, train_samples, test_samples):
        raise NotImplementedError()


class SAREMThreadedExperiment(BaseThreadedExperiment):
    def __init__(
        self, train_samples, test_samples, experiment_id, c, config, method_dir, sem
    ):
        BaseThreadedExperiment.__init__(
            self, train_samples, test_samples, experiment_id, c, config, method_dir, sem
        )

    def train_and_test(self, train_samples, test_samples):
        np.random.seed(self.experiment_id)
        torch.manual_seed(self.experiment_id)
        tf.random.set_seed(self.experiment_id)

        X_train, y_train, s_train = train_samples
        X_test, y_test, s_test = test_samples

        y_train = np.where(y_train == 1, 1, 0)
        y_test = np.where(y_test == 1, 1, 0)

        em_training_start = time.perf_counter()

        f_model, e_model, info = pu_learn_sar_em(
            X_train,
            s_train,
            range(X_train.shape[1]),
            verbose=True,
            log_prefix=f"Exp {self.experiment_id}, c: {self.c:.2f} || ",
        )

        em_training_time = time.perf_counter() - em_training_start

        # evaluate
        # propensity = np.zeros_like(s_test)
        # metrics = evaluate_all(
        #     y_test,
        #     s_test,
        #     propensity,
        #     f_model.predict_proba(X_test),
        #     e_model.predict_proba(X_test),
        # )

        y_pred = np.where(f_model.predict_proba(X_test) > 0.5, 1, 0)
        y_pred[s_test == 1] = 1

        acc = sk_metrics.accuracy_score(y_test, y_pred)
        precision = sk_metrics.precision_score(y_test, y_pred)
        recall = sk_metrics.recall_score(y_test, y_pred)
        f1_score = sk_metrics.f1_score(y_test, y_pred)

        u_acc = sk_metrics.accuracy_score(y_test[s_test == 0], y_pred[s_test == 0])
        u_precision = sk_metrics.precision_score(
            y_test[s_test == 0], y_pred[s_test == 0]
        )
        u_recall = sk_metrics.recall_score(y_test[s_test == 0], y_pred[s_test == 0])
        u_f1_score = sk_metrics.f1_score(y_test[s_test == 0], y_pred[s_test == 0])

        metric_values = {
            "Method": "SAR-EM",
            "Accuracy": acc,
            "Precision": precision,
            "Recall": recall,
            "F1 score": f1_score,
            "U-Accuracy": u_acc,
            "U-Precision": u_precision,
            "U-Recall": u_recall,
            "U-F1 score": u_f1_score,
            "Time": em_training_time,
        }

        os.makedirs(self.method_dir, exist_ok=True)
        with open(os.path.join(self.method_dir, "metric_values.json"), "w") as f:
            json.dump(metric_values, f)
