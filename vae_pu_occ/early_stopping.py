import copy

import numpy as np


class EarlyStopping:
    def __init__(self, patience=5, best_metric_type="min"):
        self.patience = patience
        self.counter = 0

        self.best_epoch = None
        self.best_model = None

        if best_metric_type == "min":
            self.best_metric_function = min
            self.best_metric = np.inf
        elif best_metric_type == "max":
            self.best_metric_function = max
            self.best_metric = -np.inf
        else:
            NotImplementedError("Invalid best_metric_type")

    def check_stop(self, epoch, epoch_metric, model):
        if epoch_metric == self.best_metric_function(epoch_metric, self.best_metric):
            self.counter = 0
            self.best_metric = epoch_metric
            self.best_epoch = epoch
            self.best_model = copy.deepcopy(model)
        else:
            self.counter += 1
            if self.counter == self.patience:
                return True
        return False
