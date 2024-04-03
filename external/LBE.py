import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Classifier(nn.Module):
    def forward(self, x):
        pass


class MLPClassifier(Classifier):
    def __init__(self, input_dim, hidden_dim=10):
        super(MLPClassifier, self).__init__()
        self.h = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x, sigmoid=False):
        h = self.h(x)
        if sigmoid:
            h = torch.sigmoid(h)
        return h


class LogisticClassifier(Classifier):
    def __init__(self, input_dim):
        super(LogisticClassifier, self).__init__()
        self.h = nn.Sequential(
            nn.Linear(input_dim, 1),
        )

    def forward(self, x, sigmoid=False):
        h = self.h(x)
        if sigmoid:
            h = torch.sigmoid(h)
        return h


class PropensityEstimator(nn.Module):
    def forward(self, x):
        pass


class MLPPropensityEstimator(PropensityEstimator):
    def __init__(self, input_dim, hidden_dim=10):
        super(MLPPropensityEstimator, self).__init__()
        self.eta = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

        for layer in [
            module for module in self.eta.modules() if isinstance(module, nn.Linear)
        ]:
            layer.weight = nn.Parameter(torch.zeros_like(layer.weight))
            layer.bias = nn.Parameter(torch.zeros_like(layer.bias))

    def forward(self, x, sigmoid=False):
        eta = self.eta(x)
        if sigmoid:
            eta = torch.sigmoid(eta)
        return eta


class LogisticPropensityEstimator(PropensityEstimator):
    def __init__(self, input_dim):
        super(LogisticPropensityEstimator, self).__init__()
        self.eta = nn.Sequential(
            nn.Linear(input_dim, 1),
        )

        for layer in [
            module for module in self.eta.modules() if isinstance(module, nn.Linear)
        ]:
            layer.weight = nn.Parameter(torch.zeros_like(layer.weight))
            layer.bias = nn.Parameter(torch.zeros_like(layer.bias))

    def forward(self, x, sigmoid=False):
        eta = self.eta(x)
        if sigmoid:
            eta = torch.sigmoid(eta)
        return eta


class LBE(nn.Module):
    def __init__(self, input_dim, kind="MLP", hidden_dim=10):
        super(LBE, self).__init__()
        if kind == "MLP":
            self.h = MLPClassifier(input_dim, hidden_dim)
            self.eta = MLPPropensityEstimator(input_dim, hidden_dim)
        elif kind == "LF":
            self.h = LogisticClassifier(input_dim)
            self.eta = LogisticPropensityEstimator(input_dim)

    def get_classifier_params(self):
        return self.h.parameters()

    def set_classifier_params(self, weights, intercept):
        state_dict = self.h.state_dict()
        state_dict["h.0.weight"] = torch.tensor(
            weights, dtype=state_dict["h.0.weight"].dtype
        )
        state_dict["h.0.bias"] = torch.tensor(
            intercept, dtype=state_dict["h.0.bias"].dtype
        )
        self.h.load_state_dict(state_dict)

    def get_theta_h(self):
        value = torch.tensor([], dtype=torch.float)
        for param in self.h.parameters():
            value = torch.cat([value.squeeze(), param.data.reshape(-1)])
        return value

    def get_theta_eta(self):
        value = torch.tensor([], dtype=torch.float)
        for param in self.eta.parameters():
            value = torch.cat([value.squeeze(), param.data.reshape(-1)])
        return value

    def forward(self, x):
        h = self.h(x, sigmoid=True)
        return h

    def predict_eta(self, x):
        eta = self.eta(x, sigmoid=True)
        return eta

    def E_step(self, x, s, y=None):
        with torch.no_grad():
            h = self.h(x, sigmoid=True).squeeze()
            eta = self.eta(x, sigmoid=True).squeeze()

            P_y_hat_1 = torch.where(s == 1, eta, 1 - eta) * h
            P_y_hat_0 = torch.where(s == 1, 0, 1) * (1 - h)

            # Fix for a rare case where real y = 1, predicted h(x) = 0.
            # In that case P_y_hat_1 and P_y_hat_0 are 0, so later normalization
            # returns NaN.
            # P(y_i = 1|x_i, s_i) should be 1 in that case, introducing
            # non-zero value will fix that.
            EPS = torch.tensor(1e-5).to(P_y_hat_1.device)
            P_y_hat_1 = torch.where((h == 0) & (s == 1), EPS, P_y_hat_1)
            # Analogously, when predicted h(x) = 1 and eta(x) = 1, but s = 0.
            # P(y_i = 0|x_i, s_i) should be 1 in that case.
            P_y_hat_0 = torch.where((h == 1) & (eta == 1) & (s == 0), EPS, P_y_hat_0)

            P_y_hat = torch.cat(
                [P_y_hat_0.reshape(-1, 1), P_y_hat_1.reshape(-1, 1)], axis=1
            )
            P_y_hat /= P_y_hat.sum(axis=1).reshape(-1, 1)
            return P_y_hat

    def loss(self, x, s, P_y_hat):
        h = self.h(x).squeeze()
        eta = self.eta(x).squeeze()

        log_h = F.logsigmoid(h)
        log_1_minus_h = F.logsigmoid(-h)
        log_eta = F.logsigmoid(eta)
        log_1_minus_eta = F.logsigmoid(-eta)

        loss = torch.where(
            s == 1,
            P_y_hat[:, 1] * (log_h + log_eta)
            + P_y_hat[:, 0] * (log_1_minus_h + log_eta),
            P_y_hat[:, 1] * (log_h + log_1_minus_eta)
            + P_y_hat[:, 0] * (log_1_minus_h + log_1_minus_eta),
        )

        with torch.no_grad():
            x = torch.cat([x, torch.ones((len(x), 1)).cuda()], dim=1)
            sigma_h = torch.sigmoid(h)
            sigma_eta = torch.sigmoid(eta)
            grad_theta_1 = (
                (P_y_hat[:, 0] * sigma_h).reshape(-1, 1) * x
                + (P_y_hat[:, 1] * (sigma_h - 1)).reshape(-1, 1) * x
            ).sum(axis=0)
            grad_theta_2 = (
                (
                    (-1) ** (s + 1)
                    * (
                        1
                        * P_y_hat[:, 1]
                        / torch.where(s == 1, sigma_eta, 1 - sigma_eta)
                    )
                    * sigma_eta
                    * (sigma_eta - 1)
                ).reshape(-1, 1)
                * x
            ).sum(axis=0)

        return -torch.sum(loss), grad_theta_1, grad_theta_2

    def pre_train(self, x, s, epochs=100, lr=1e-3, print_msg=False):
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(self.h.parameters(), lr=lr)

        for epoch in range(epochs):
            optimizer.zero_grad()

            # Forward pass
            s_logits = self.h(x)
            # Compute Loss
            loss = criterion(s_logits.squeeze(), s)
            # Backward pass
            loss.backward()
            optimizer.step()

            if print_msg:
                print("Epoch {}: train loss: {}".format(epoch, loss.item()))


class LBE_alternative(nn.Module):
    def __init__(self, input_dim):
        super(LBE_alternative, self).__init__()
        self.input_dim = input_dim
        self.theta_h = nn.Parameter(
            torch.randn((input_dim + 1, 1)) / 100, requires_grad=True
        )
        self.theta_eta = nn.Parameter(
            torch.randn((input_dim + 1, 1)) / 100, requires_grad=True
        )

    def __init_param__(self):
        weight = nn.Parameter(torch.empty((self.input_dim, 1)))
        bias = nn.Parameter(torch.empty(1))

        nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
        if bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(bias, -bound, bound)

        return nn.Parameter(
            torch.cat([weight.reshape(self.input_dim, 1), bias.reshape(1, 1)]),
            requires_grad=True,
        )

    def get_clasifier_params(self):
        return [self.theta_h]

    def h(self, x, sigmoid=False):
        x = torch.cat([x, torch.ones((len(x), 1)).to("cuda")], dim=1)
        h = x @ self.theta_h
        if sigmoid:
            h = torch.sigmoid(h)
        return h

    def eta(self, x, sigmoid=False):
        x = torch.cat([x, torch.ones((len(x), 1)).to("cuda")], dim=1)
        eta = x @ self.theta_eta
        if sigmoid:
            eta = torch.sigmoid(eta)
        return eta

    def get_theta_h(self):
        return self.theta_h.data.squeeze()

    def get_theta_eta(self):
        return self.theta_eta.data.squeeze()

    def forward(self, x):
        h = self.h(x, sigmoid=True)
        return h

    def predict_eta(self, x):
        eta = self.eta(x, sigmoid=True)
        return eta

    def E_step(self, x, s):
        with torch.no_grad():
            h = self.h(x, sigmoid=True).squeeze()
            eta = self.eta(x, sigmoid=True).squeeze()

            P_y_hat_1 = torch.where(s == 1, eta, 1 - eta) * h
            P_y_hat_0 = torch.where(s == 1, 0, 1) * (1 - h)

            P_y_hat = torch.cat(
                [P_y_hat_0.reshape(-1, 1), P_y_hat_1.reshape(-1, 1)], axis=1
            )
            P_y_hat /= P_y_hat.sum(axis=1).reshape(-1, 1)
            return P_y_hat

    def loss(self, x, s, P_y_hat):
        h = self.h(x).squeeze()
        eta = self.eta(x).squeeze()

        log_h = F.logsigmoid(h)
        log_1_minus_h = F.logsigmoid(-h)
        log_eta = F.logsigmoid(eta)
        log_1_minus_eta = F.logsigmoid(-eta)

        loss = torch.where(
            s == 1,
            P_y_hat[:, 1] * (log_h + log_eta) + 0,
            P_y_hat[:, 1] * (log_h + log_1_minus_eta) + P_y_hat[:, 0] * log_1_minus_h,
        )

        with torch.no_grad():
            x = torch.cat([x, torch.ones((len(x), 1)).to("cuda")], dim=1)
            sigma_h = torch.sigmoid(h)
            sigma_eta = torch.sigmoid(eta)
            grad_theta_1 = (
                (P_y_hat[:, 0] * sigma_h).reshape(-1, 1) * x
                + (P_y_hat[:, 1] * (sigma_h - 1)).reshape(-1, 1) * x
            ).sum(axis=0)
            grad_theta_2 = (
                (
                    (-1) ** (s + 1)
                    * (
                        1
                        * P_y_hat[:, 1]
                        / torch.where(s == 1, sigma_eta, 1 - sigma_eta)
                    )
                    * sigma_eta
                    * (sigma_eta - 1)
                ).reshape(-1, 1)
                * x
            ).sum(axis=0)

        return -torch.sum(loss), grad_theta_1, grad_theta_2

    def pre_train(self, x, s, epochs=100, lr=1e-3, print_msg=False):
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam([self.theta_h], lr=lr)

        for epoch in range(epochs):
            optimizer.zero_grad()

            # Forward pass
            s_logits = self.h(x)
            # Compute Loss
            loss = criterion(s_logits.squeeze(), s)
            # Backward pass
            loss.backward()
            optimizer.step()

            if print_msg:
                print("Epoch {}: train loss: {}".format(epoch, loss.item()))


import copy

import torch


class EarlyStopping:
    def __init__(self, early_stopping_epochs, best_metric_type="min"):
        self.best_metric_type = best_metric_type
        if self.best_metric_type == "max":
            self.best_metric_value = -torch.inf
        elif self.best_metric_type == "min":
            self.best_metric_value = torch.inf

        self.counter = 0
        self.early_stopping_epochs = early_stopping_epochs
        self.best_model = None

    def is_better(self, metric_value):
        return (
            self.best_metric_type == "max" and metric_value > self.best_metric_value
        ) or (self.best_metric_type == "min" and metric_value < self.best_metric_value)

    def check_stopping(self, metric_value, model):
        if self.is_better(metric_value):
            self.counter = 0
            self.best_metric_value = metric_value
            self.best_model = copy.deepcopy(model)
            return False
        else:
            self.counter += 1
            if self.counter == self.early_stopping_epochs:
                return True

    def get_best_metric(self):
        return self.best_metric_value

    def get_best_model(self):
        return copy.deepcopy(self.best_model)


import numpy as np
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader, TensorDataset


def convert_to_DL(samples):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    x, y, s = samples
    y = np.where(y == 1, 1, 0)
    x, y, s = (
        torch.from_numpy(x).float().to(device),
        torch.from_numpy(y).float().to(device),
        torch.from_numpy(s).float().to(device),
    )

    shuffled_indices = torch.randperm(y.shape[0])
    x, y, s = x[shuffled_indices], y[shuffled_indices], s[shuffled_indices]

    ds = TensorDataset(x, y, s)
    dl = DataLoader(ds, batch_size=64, shuffle=False)
    return dl


def train_LBE(
    train_samples,
    val_samples,
    kind="LF",
    pretraining_epochs=100,
    pretraining_lr=1e-2,
    training_max_epochs=100,
    training_m_step_max_epochs=100,
    training_lr=1e-3,
    iter_early_stopping_max_epochs=2,
    M_step_early_stopping_max_epochs=100,
    verbose=False,
    log_prefix="",
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x_train, _, s_train = train_samples

    DL_train = convert_to_DL(train_samples)
    DL_val = convert_to_DL(val_samples)

    n_inputs = x_train.shape[1]
    lbe = LBE(n_inputs, kind=kind).to(device)

    # PRETRAINING
    if verbose:
        print(f"{log_prefix}Pretraining started...")
    clf = LogisticRegression()
    clf.fit(x_train, s_train)
    lbe.set_classifier_params(clf.coef_, clf.intercept_)
    if verbose:
        print(f"{log_prefix}Pretraining finished")

    # TRAINING
    binary_loss = nn.BCELoss()
    optimizer = optim.Adam(lbe.parameters(), lr=training_lr)
    lbe_early_stopping = EarlyStopping(
        iter_early_stopping_max_epochs, best_metric_type="min"
    )
    for epoch in range(training_max_epochs):
        epoch_early_stopping = EarlyStopping(
            M_step_early_stopping_max_epochs, best_metric_type="min"
        )
        P_y_hats = []
        for x, _, s in DL_train:
            P_y_hat = lbe.E_step(x, s)
            P_y_hats.append(P_y_hat)

        for M_step_iter in range(training_m_step_max_epochs):
            i = 0
            for x, _, s in DL_train:
                P_y_hat = P_y_hats[i]
                i += 1

                optimizer.zero_grad()

                # Forward pass
                loss, _, _ = lbe.loss(x, s, P_y_hat)
                # Backward pass
                loss.backward()

                optimizer.step()

            with torch.no_grad():

                def get_metrics(DL):
                    ys = []
                    y_probas = []
                    y_preds = []
                    eta_preds = []
                    losses = []

                    for x, y, s in DL:
                        y_probas.append(lbe(x).reshape(-1).detach().cpu().numpy())
                        y_preds.append(
                            torch.where(lbe(x) > 0.5, 1, 0)
                            .reshape(-1)
                            .detach()
                            .cpu()
                            .numpy()
                        )
                        eta_preds.append(
                            torch.where(lbe.predict_eta(x) > 0.5, 1, 0)
                            .reshape(-1)
                            .detach()
                            .cpu()
                            .numpy()
                        )
                        ys.append(y.detach().cpu().numpy())

                        P_y_hat = lbe.E_step(x, s)
                        loss, _, _ = lbe.loss(x, s, P_y_hat)
                        losses.append(loss.detach().cpu().numpy())

                    y_pred = np.concatenate(y_preds)
                    y_proba = np.concatenate(y_probas)
                    eta_pred = np.concatenate(eta_preds)
                    y = np.concatenate(ys)

                    acc = metrics.accuracy_score(y, y_pred)
                    auc = metrics.roc_auc_score(y, y_proba)
                    loss = np.mean(losses)
                    return loss, acc, auc

                train_loss, train_acc, train_auc = get_metrics(DL_train)
                val_loss, val_acc, val_auc = get_metrics(DL_val)

            if epoch_early_stopping.check_stopping(val_loss, lbe):
                val_loss = epoch_early_stopping.get_best_metric()
                lbe = epoch_early_stopping.get_best_model()
                break

        if verbose:
            print(
                f"{log_prefix}Epoch {epoch}: loss {train_loss:.4f}, acc {train_acc*100:.2f}, val_loss {val_loss:.4f}, val_acc {val_acc*100:.2f}"
            )

        if lbe_early_stopping.check_stopping(val_loss, lbe):
            val_loss = lbe_early_stopping.get_best_metric()
            lbe = lbe_early_stopping.get_best_model()
            if verbose:
                print(f"{log_prefix}Early stopping - best val loss: {loss}")
            break

    return lbe


def predict_LBE(lbe, test_samples):
    DL_test = convert_to_DL(test_samples)

    with torch.no_grad():
        ys = []
        ss = []
        y_probas = []
        eta_probas = []

        for x, y, s in DL_test:
            y_probas.append(lbe(x).reshape(-1).detach().cpu().numpy())
            eta_probas.append(lbe.predict_eta(x).reshape(-1).detach().cpu().numpy())
            ys.append(y.detach().cpu().numpy())
            ss.append(s.detach().cpu().numpy())

    y = np.concatenate(ys)
    y_proba = np.concatenate(y_probas)

    s = np.concatenate(ss)
    eta_proba = np.concatenate(eta_probas)
    s_proba = y_proba * eta_proba
    return y_proba, y, s_proba, s


# def eval_LBE(lbe, test_samples, verbose=False, log_prefix="", use_s_rule=False):
#     y_proba, y, s_proba, s = predict_LBE(lbe, test_samples)

#     if use_s_rule:
#         y_pred = np.where(s == 1, 1, np.where(y_proba > (1 + s_proba) / 2, 1, 0))
#     else:
#         y_pred = np.where(s == 1, 1, np.where(y_proba > 0.5, 1, 0))

#     accuracy = metrics.accuracy_score(y, y_pred)
#     precision = metrics.precision_score(y, y_pred)
#     recall = metrics.recall_score(y, y_pred)
#     f1 = metrics.f1_score(y, y_pred)
#     u_accuracy = metrics.accuracy_score(y[s == 0], y_pred[s == 0])
#     u_precision = metrics.precision_score(y[s == 0], y_pred[s == 0])
#     u_recall = metrics.recall_score(y[s == 0], y_pred[s == 0])
#     u_f1 = metrics.f1_score(y[s == 0], y_pred[s == 0])

#     if verbose:
#         print(f"{log_prefix}LBE accuracy: {100 * accuracy:.2f}%")
#         # print(f"{log_prefix}LBE precision: {100 * precision:.2f}%")
#         # print(f"{log_prefix}LBE recall: {100 * recall:.2f}%")
#         print(f"{log_prefix}LBE F1-score: {100 * f1:.2f}%")
#         print(f"{log_prefix}LBE U-accuracy: {100 * u_accuracy:.2f}%")
#         # print(f"{log_prefix}LBE U-precision: {100 * u_precision:.2f}%")
#         # print(f"{log_prefix}LBE U-recall: {100 * u_recall:.2f}%")
#         print(f"{log_prefix}LBE U-F1-score: {100 * u_f1:.2f}%")

#     return accuracy, precision, recall, f1, u_accuracy, u_precision, u_recall, u_f1
