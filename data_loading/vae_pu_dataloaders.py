# %%
import os

import numba as nb
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from torchvision.datasets import CIFAR10, MNIST, STL10
from ucimlrepo import fetch_ucirepo

from data_loading.small_dataset_wrapper import DATASET_NAMES, get_small_dataset


@nb.jit(parallel=True)
def is_in_set_pnb(a, b):
    shape = a.shape
    a = a.ravel()
    n = len(a)
    result = np.full(n, False)
    for i in nb.prange(n):
        if a[i] in b:
            result[i] = True
    return result.reshape(shape)


def __get_pu_mnist(
    included_classes,
    positive_classes,
    case_control,
    scar,
    label_frequency,
    data_dir="./data/",
    val_size=0.15,
    test_size=0.15,
    synthetic_labels=False,
    optimize_intercept_only=False,
    max_sampling=True,
):
    included_classes = torch.tensor(included_classes)
    positive_classes = torch.tensor(positive_classes)

    # Prepare datasets
    train_base = MNIST(root=data_dir, train=True, download=True)
    test_base = MNIST(root=data_dir, train=False, download=True)

    X = torch.cat(
        [
            train_base.data.float().view(-1, 784) / 255.0,
            test_base.data.float().view(-1, 784) / 255.0,
        ],
        0,
    )
    y = torch.cat([train_base.targets, test_base.targets], 0)

    indices_all = torch.where(torch.isin(y, included_classes))
    X, y = X[indices_all], y[indices_all]

    if not scar and not synthetic_labels:
        o = torch.zeros_like(y)
        for pos_class in positive_classes:
            # Get positive distribution
            p_indices = torch.where(y == pos_class)[0]
            X_pos = X[p_indices]

            # Sample positive distribution
            num_labeled = int(len(X_pos) * label_frequency)

            boldness = X_pos.sum(axis=1)
            if max_sampling:
                bold_sort = torch.argsort(boldness, descending=True)
                most_bold_pos_indices = bold_sort[0:num_labeled]
                l_indices = p_indices[most_bold_pos_indices]
            else:
                idx = boldness.multinomial(num_samples=num_labeled, replacement=False)
                l_indices = p_indices[idx]

            o[l_indices] = 1

    X = 2 * X - 1
    y = torch.where(torch.isin(y, positive_classes), 1, -1)
    if scar:
        o = get_scar_labels(y, label_frequency)
        g, intercept = None, None
    elif synthetic_labels:
        o, g, intercept = get_synthetic_labels(
            X, y, label_frequency, optimize_intercept_only
        )
    else:
        g, intercept = None, None

    return (
        *split_datasets(
            X, y, o, label_frequency, val_size, test_size, case_control=case_control
        ),
        g,
        intercept,
    )


def __get_pu_cifar10_precomputed(
    included_classes,
    positive_classes,
    case_control,
    scar,
    label_frequency,
    data_dir="./data/",
    val_size=0.15,
    test_size=0.15,
    synthetic_labels=False,
    optimize_intercept_only=False,
    max_sampling=True,
):
    included_classes = torch.tensor(included_classes)
    positive_classes = torch.tensor(positive_classes)

    X_preprocessed = torch.load(
        os.path.join(data_dir, "cifar10.pt"), map_location="cpu"
    )

    # Prepare datasets
    train_base = CIFAR10(root=data_dir, train=True, download=True)
    test_base = CIFAR10(root=data_dir, train=False, download=True)

    X = torch.cat(
        [
            torch.tensor(train_base.data).float().reshape(-1, 32, 32, 3) / 255.0,
            torch.tensor(test_base.data).float().reshape(-1, 32, 32, 3) / 255.0,
        ],
        0,
    )
    y = torch.cat(
        [torch.tensor(train_base.targets), torch.tensor(test_base.targets)], 0
    )

    indices_all = torch.where(torch.isin(y, included_classes))
    X, y, X_preprocessed = X[indices_all], y[indices_all], X_preprocessed[indices_all]

    if not scar and not synthetic_labels:
        o = torch.zeros_like(y)
        for pos_class in positive_classes:
            # Get positive distribution
            p_indices = torch.where(y == pos_class)[0]
            X_pos = X[p_indices]

            # Sample positive distribution
            num_labeled = int(len(X_pos) * label_frequency)

            redness = (
                (X_pos[:, :, :, 0] - X_pos[:, :, :, 1])
                + (X_pos[:, :, :, 0] - X_pos[:, :, :, 1])
            ) / 2

            redness = redness.reshape(-1, 32 * 32).sum(axis=1)

            if max_sampling:
                red_sort = torch.argsort(redness, descending=True)
                most_red_pos_indices = red_sort[0:num_labeled]
                l_indices = p_indices[most_red_pos_indices]
            else:
                # shift to positive
                redness += -torch.min(redness)

                idx = redness.multinomial(num_samples=num_labeled, replacement=False)
                l_indices = p_indices[idx]

            o[l_indices] = 1

    X = X_preprocessed
    y = torch.where(torch.isin(y, positive_classes), 1, -1)
    if scar:
        o = get_scar_labels(y, label_frequency)
        g, intercept = None, None
    elif synthetic_labels:
        o, g, intercept = get_synthetic_labels(
            X, y, label_frequency, optimize_intercept_only
        )
    else:
        g, intercept = None, None

    return (
        *split_datasets(
            X, y, o, label_frequency, val_size, test_size, case_control=case_control
        ),
        g,
        intercept,
    )


def __get_pu_stl_precomputed(
    included_classes,
    positive_classes,
    case_control,
    scar,
    label_frequency,
    data_dir="./data/",
    val_size=0.15,
    test_size=0.15,
    synthetic_labels=False,
    optimize_intercept_only=False,
    max_sampling=True,
):
    included_classes = torch.tensor(included_classes)
    positive_classes = torch.tensor(positive_classes)

    X_preprocessed = torch.load(os.path.join(data_dir, "stl10.pt"), map_location="cpu")

    # Prepare datasets
    train_base = STL10(root=data_dir, split="train", download=True)
    test_base = STL10(root=data_dir, split="test", download=True)

    X = torch.cat(
        [
            torch.tensor(train_base.data).float().reshape(-1, 32, 32, 3) / 255.0,
            torch.tensor(test_base.data).float().reshape(-1, 32, 32, 3) / 255.0,
        ],
        0,
    )
    y = torch.cat([torch.tensor(train_base.labels), torch.tensor(test_base.labels)], 0)

    indices_all = torch.where(torch.isin(y, included_classes))
    X, y, X_preprocessed = X[indices_all], y[indices_all], X_preprocessed[indices_all]

    if not scar and not synthetic_labels:
        o = torch.zeros_like(y)
        for pos_class in positive_classes:
            # Get positive distribution
            p_indices = torch.where(y == pos_class)[0]
            X_pos = X[p_indices]

            # Sample positive distribution
            num_labeled = int(len(X_pos) * label_frequency)

            redness = (
                (X_pos[:, :, :, 0] - X_pos[:, :, :, 1])
                + (X_pos[:, :, :, 0] - X_pos[:, :, :, 1])
            ) / 2
            redness = redness.reshape(-1, 32 * 32).sum(axis=1)

            if max_sampling:
                red_sort = torch.argsort(redness, descending=True)
                most_red_pos_indices = red_sort[0:num_labeled]
                l_indices = p_indices[most_red_pos_indices]
            else:
                # shift to positive
                redness += -torch.min(redness)

                idx = redness.multinomial(num_samples=num_labeled, replacement=False)
                l_indices = p_indices[idx]

            o[l_indices] = 1

    X = X_preprocessed
    y = torch.where(torch.isin(y, positive_classes), 1, -1)
    if scar:
        o = get_scar_labels(y, label_frequency)
        g, intercept = None, None
    elif synthetic_labels:
        o, g, intercept = get_synthetic_labels(
            X, y, label_frequency, optimize_intercept_only
        )
    else:
        g, intercept = None, None

    return (
        *split_datasets(
            X, y, o, label_frequency, val_size, test_size, case_control=case_control
        ),
        g,
        intercept,
    )


def get_xs_data(
    label_frequency,
    n_samples,
    n_features,
    val_size=0.15,
    test_size=0.15,
    case_control=False,
    scar=False,
    propensity_type="logistic",  # 'logistic', 'cauchy', 'logistic^10'
):
    from torch.distributions.multivariate_normal import MultivariateNormal

    if "separated" in propensity_type:
        mu = 3 / torch.arange(1, n_features + 1)
    else:
        mu = 1 / torch.arange(1, n_features + 1)

    X = torch.cat(
        [
            MultivariateNormal(torch.zeros(n_features), torch.eye(n_features)).sample(
                (n_samples // 2,)
            ),
            MultivariateNormal(mu, torch.eye(n_features)).sample((n_samples // 2,)),
        ]
    )
    idx = torch.randperm(X.shape[0])
    X = X[idx]

    pi = 0.5

    def calculate_y_proba(X):
        if type(X) == np.ndarray:
            X = torch.from_numpy(X)
        X1 = torch.hstack([X, torch.ones((X.shape[0], 1))])

        beta = torch.cat(
            [mu, (np.log(pi / (1 - pi)) - 0.5 * mu.norm() ** 2).reshape(-1)]
        )

        y_proba = torch.sigmoid(X1 @ beta)
        return y_proba

    def calculate_y_proba_diagonal(X):
        if type(X) == np.ndarray:
            X = torch.from_numpy(X)
        X1 = torch.hstack([X, torch.ones((X.shape[0], 1))])

        cov = torch.diag(torch.tensor([1, 2]).repeat(X.shape[1] // 2)).float()
        inv_cov = cov.inverse()

        beta = torch.cat(
            [
                mu @ inv_cov,
                (np.log(pi / (1 - pi)) - 0.5 * mu @ inv_cov @ mu.T).reshape(-1),
            ]
        )

        y_proba = torch.sigmoid(X1 @ beta)
        return y_proba

    def calculate_propensity_logistic(X, y, propensity_type, y_proba, gamma):
        if type(X) == np.ndarray:
            X = torch.from_numpy(X)
        if type(y) == np.ndarray:
            y = torch.from_numpy(y)
        X1 = torch.hstack([X, torch.ones((X.shape[0], 1))])

        if "logistic" in propensity_type or "diagonal" in propensity_type:
            propensity = torch.sigmoid(X1 @ gamma)
        if "^10" in propensity_type:
            propensity = propensity**10
        return propensity

    def calculate_propensity_SCAR(X, y, propensity_type, y_proba, gamma):
        if type(X) == np.ndarray:
            X = torch.from_numpy(X)

        eta = label_frequency * torch.ones(len(X))
        return eta

    if propensity_type == "1-2-diagonal":
        y_proba = calculate_y_proba_diagonal(X)
    else:
        y_proba = calculate_y_proba(X)

    y = torch.where(torch.bernoulli(y_proba) == 1, 1, -1)

    calculate_propensity = None
    if scar:
        calculate_propensity = calculate_propensity_SCAR
        gamma = None
        propensity = calculate_propensity(X, y, propensity_type, y_proba, gamma)
        s_proba = y_proba * propensity

        s = torch.where(
            y == 1,
            torch.bernoulli(propensity),
            torch.tensor(0, dtype=torch.float),
        )
        g, intercept = None, None
    else:
        from scipy.optimize import differential_evolution

        X1 = torch.hstack([X, torch.ones((X.shape[0], 1))])

        def get_gamma(g, intercept):
            gamma = g * torch.zeros(n_features)
            if "inverse" in propensity_type:
                gamma[0] = -1
            else:
                gamma[0] = 1

            return torch.cat([gamma, torch.tensor(intercept).reshape(-1).float()])

        def tune_gamma_g_and_intercept_to_lf(param):
            g, intercept = param[0], param[1]
            gamma = get_gamma(g, intercept)
            propensity = calculate_propensity_logistic(
                X, y, propensity_type, y_proba, gamma
            )

            lf_approx = torch.mean(propensity[y == 1])
            return torch.abs(lf_approx - label_frequency)

        def tune_gamma_intercept_only_to_lf(param):
            intercept = param
            gamma = get_gamma(0.5, intercept)
            propensity = calculate_propensity_logistic(
                X, y, propensity_type, y_proba, gamma
            )

            lf_approx = torch.mean(propensity[y == 1])
            return torch.abs(lf_approx - label_frequency)

        if "interceptonly" in propensity_type:
            res = differential_evolution(
                tune_gamma_intercept_only_to_lf,
                bounds=[(-n_features, n_features)],
            )
            g = 0.5
            intercept = res.x
        else:
            res = differential_evolution(
                tune_gamma_g_and_intercept_to_lf,
                bounds=[(0, 1), (-n_features, n_features)],
            )
            g, intercept = res.x

        gamma = get_gamma(g, intercept)
        calculate_propensity = calculate_propensity_logistic
        propensity = calculate_propensity(X, y, propensity_type, y_proba, gamma)
        s = torch.bernoulli(propensity)
        s = torch.where(y == 1, s, torch.tensor(0, dtype=torch.float))
        s_proba = y_proba * propensity

    (
        train_samples,
        val_samples,
        test_samples,
        label_frequency,
        pi_p,
        n_input,
    ) = split_datasets(
        X, y, s, label_frequency, val_size, test_size, case_control=case_control
    )

    y_proba_test = calculate_y_proba(test_samples[0])
    propensity_test = calculate_propensity(
        test_samples[0], test_samples[1], propensity_type, y_proba_test, gamma
    )
    s_proba_test = y_proba_test * propensity_test

    return (
        train_samples,
        val_samples,
        test_samples,
        label_frequency,
        pi_p,
        n_input,
        g,
        intercept,
        y_proba_test,
        s_proba_test,
    )


def get_xs_data_v2(
    label_frequency,
    n_samples,
    n_features,
    val_size=0.15,
    test_size=0.15,
    case_control=False,
    scar=False,
    propensity_type="logistic",
):
    from torch.distributions.multivariate_normal import MultivariateNormal

    mu = 1 / torch.arange(1, n_features + 1)

    if "1-2-diagonal" in propensity_type:
        cov = torch.diag(torch.tensor([1, 2]).repeat(n_features / 2)).float()
    else:
        cov = torch.diag(torch.tensor([1]).repeat(n_features)).float()

    X = torch.cat(
        [
            MultivariateNormal(torch.zeros(n_features), cov).sample((n_samples // 2,)),
            MultivariateNormal(mu, cov).sample((n_samples // 2,)),
        ]
    )
    idx = torch.randperm(X.shape[0])
    X = X[idx]

    pi = 0.5

    def calculate_y_proba(X):
        if type(X) == np.ndarray:
            X = torch.from_numpy(X)
        X1 = torch.hstack([X, torch.ones((X.shape[0], 1))])

        beta = torch.cat(
            [mu, (np.log(pi / (1 - pi)) - 0.5 * mu.norm() ** 2).reshape(-1)]
        )

        y_proba = torch.sigmoid(X1 @ beta)
        return y_proba

    def calculate_y_proba_diagonal(X):
        if type(X) == np.ndarray:
            X = torch.from_numpy(X)
        X1 = torch.hstack([X, torch.ones((X.shape[0], 1))])

        inv_cov = cov.inverse()

        beta = torch.cat(
            [
                mu @ inv_cov,
                (np.log(pi / (1 - pi)) - 0.5 * mu @ inv_cov @ mu.T).reshape(-1),
            ]
        )

        y_proba = torch.sigmoid(X1 @ beta)
        return y_proba

    def calculate_propensity_logistic(X, y, propensity_type, y_proba, gamma):
        if type(X) == np.ndarray:
            X = torch.from_numpy(X)
        if type(y) == np.ndarray:
            y = torch.from_numpy(y)
        X1 = torch.hstack([X, torch.ones((X.shape[0], 1))])

        if "logistic" in propensity_type or "diagonal" in propensity_type:
            propensity = torch.sigmoid(X1 @ gamma)
        if "inverse" in propensity_type:
            propensity = 1 - propensity
        if "^10" in propensity_type:
            propensity = propensity**10
        return propensity

    def calculate_propensity_SCAR(X, y, propensity_type, y_proba, gamma):
        if type(X) == np.ndarray:
            X = torch.from_numpy(X)

        eta = label_frequency * torch.ones(len(X))
        return eta

    if "1-2-diagonal" in propensity_type:
        y_proba = calculate_y_proba_diagonal(X)
    else:
        y_proba = calculate_y_proba(X)

    y = torch.where(torch.bernoulli(y_proba) == 1, 1, -1)

    calculate_propensity = None
    if scar:
        calculate_propensity = calculate_propensity_SCAR
        gamma = None
        propensity = calculate_propensity(X, y, propensity_type, y_proba, gamma)
        s_proba = y_proba * propensity

        s = torch.where(
            y == 1,
            torch.bernoulli(propensity),
            torch.tensor(0, dtype=torch.float),
        )
        beta_star, intercept = None, None
    else:
        from scipy.optimize import differential_evolution

        def get_logistic_beta_star(X, y):
            from sklearn.linear_model import LogisticRegression

            if type(X) == np.ndarray:
                X = torch.from_numpy(X)

            clf = LogisticRegression()
            clf.fit(X.numpy(), y.numpy())
            beta_star = clf.coef_
            intercept = clf.intercept_
            return beta_star

        beta_star = get_logistic_beta_star(X, y)

        def get_gamma(beta_star, intercept):
            return torch.cat(
                [
                    torch.tensor(beta_star).reshape(-1).float(),
                    torch.tensor(intercept).reshape(-1).float(),
                ]
            )

        def tune_gamma_intercept_only_to_lf(param):
            intercept = param
            gamma = get_gamma(beta_star, intercept)
            propensity = calculate_propensity_logistic(
                X, y, propensity_type, y_proba, gamma
            )

            lf_approx = torch.mean(propensity[y == 1])
            return torch.abs(lf_approx - label_frequency)

        res = differential_evolution(
            tune_gamma_intercept_only_to_lf,
            bounds=[(-n_features, n_features)],
        )
        intercept = res.x

        gamma = get_gamma(beta_star, intercept)
        calculate_propensity = calculate_propensity_logistic
        propensity = calculate_propensity(X, y, propensity_type, y_proba, gamma)
        s = torch.bernoulli(propensity)
        s = torch.where(y == 1, s, torch.tensor(0, dtype=torch.float))
        s_proba = y_proba * propensity

    (
        train_samples,
        val_samples,
        test_samples,
        label_frequency,
        pi_p,
        n_input,
    ) = split_datasets(
        X, y, s, label_frequency, val_size, test_size, case_control=case_control
    )

    y_proba_test = calculate_y_proba(test_samples[0])
    propensity_test = calculate_propensity(
        test_samples[0], test_samples[1], propensity_type, y_proba_test, gamma
    )
    s_proba_test = y_proba_test * propensity_test

    return (
        train_samples,
        val_samples,
        test_samples,
        label_frequency,
        pi_p,
        n_input,
        beta_star,
        intercept,
        y_proba_test,
        s_proba_test,
    )


def get_scar_labels(y, label_frequency):
    labeling_condition = (
        torch.rand_like(y, dtype=float) < label_frequency
    )  # label without bias

    o = torch.where(
        y == 1,
        torch.where(labeling_condition, 1, 0),
        0,
    )
    return o


def get_synthetic_labels(X, y, label_frequency, optimize_intercept_only):
    from scipy.optimize import differential_evolution
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression()
    clf.fit(X, y)
    beta_star = torch.tensor(clf.coef_).reshape(-1).float()

    def calculate_propensity(X, gamma):
        X1 = torch.hstack([X, torch.ones((X.shape[0], 1))])
        propensity = torch.sigmoid(X1 @ gamma)
        return propensity

    def get_gamma(g, intercept):
        gamma = g * beta_star
        return torch.cat([gamma, torch.tensor(intercept).reshape(-1).float()])

    def tune_gamma_g_and_intercept_to_lf(param):
        g, intercept = param[0], param[1]
        gamma = get_gamma(g, intercept)
        propensity = calculate_propensity(X, gamma)

        lf_approx = torch.mean(propensity[y == 1])
        return torch.abs(lf_approx - label_frequency)

    def tune_gamma_intercept_only_to_lf(param):
        g, intercept = 0.5, param
        gamma = get_gamma(g, intercept)
        propensity = calculate_propensity(X, gamma)

        lf_approx = torch.mean(propensity[y == 1])
        return torch.abs(lf_approx - label_frequency)

    max_intercept = 1 / 8
    tolerance = 0.001
    error = np.inf
    while error > tolerance:
        if optimize_intercept_only:
            res = differential_evolution(
                tune_gamma_intercept_only_to_lf,
                bounds=[(-max_intercept, max_intercept)],
            )
            g, intercept = 0.5, res.x
            error = res.fun

            max_intercept *= 2
            tolerance += 0.001
        else:
            res = differential_evolution(
                tune_gamma_g_and_intercept_to_lf,
                bounds=[
                    (0, 1),
                    (-max_intercept, max_intercept),
                ],
            )
            g, intercept = res.x
            error = res.fun

        max_intercept *= 2
        tolerance += 0.001

    gamma = get_gamma(g, intercept)
    propensity = calculate_propensity(X, gamma)
    s = torch.bernoulli(propensity)
    s = torch.where(y == 1, s, torch.tensor(0, dtype=torch.float))

    return s, g, intercept


def split_datasets(X, y, o, label_frequency, val_size, test_size, case_control=False):
    # Split datasets
    n = X.shape[0]
    n_val = int(val_size * n)
    n_test = int(test_size * n)
    n_train = n - n_val - n_test

    shuffled_indices = torch.randperm(X.shape[0])
    train_indices = shuffled_indices[:n_train]
    val_indices = shuffled_indices[n_train : (n_train + n_val)]
    test_indices = shuffled_indices[(n_train + n_val) :]

    x_train, y_train, o_train = X[train_indices], y[train_indices], o[train_indices]
    x_val, y_val, o_val = X[val_indices], y[val_indices], o[val_indices]
    x_test, y_test, o_test = X[test_indices], y[test_indices], o[test_indices]

    pi_p = (y == 1).float().mean().numpy()
    label_frequency = (o == 1).float().mean().numpy() / pi_p
    n_input = X.shape[1]

    val = x_val.numpy(), y_val.numpy(), o_val.numpy()
    test = x_test.numpy(), y_test.numpy(), o_test.numpy()

    if case_control:
        l_indices = torch.where(o_train == 1)[0]

        u_sampling_condition = (
            torch.rand_like(o_train, dtype=float) < 1 - label_frequency
        )
        is_u_sample = torch.where(
            u_sampling_condition,
            True,
            False,
        )

        x_train = torch.cat([x_train[l_indices], x_train[is_u_sample]])
        y_train = torch.cat([y_train[l_indices], y_train[is_u_sample]])
        o_train = torch.cat(
            [
                torch.ones_like(l_indices),
                torch.zeros(is_u_sample.int().sum().item()),
            ]
        )

        idx = torch.randperm(x_train.shape[0])
        x_train = x_train[idx]
        y_train = y_train[idx]
        o_train = o_train[idx]

    train = x_train.numpy(), y_train.numpy(), o_train.numpy()

    return train, val, test, label_frequency, pi_p, n_input


def get_dataset(
    name,
    device,
    label_frequency,
    data_dir="./data/",
    val_size=0.15,
    test_size=0.15,
    case_control=False,
    use_scar_labeling=False,
    synthetic_labels=False,
    optimize_intercept_only=False,
    max_sampling=True,
):
    if "MNIST" in name:
        if "3v5" in name:
            included_classes = [3, 5]
            positive_classes = [3]
        elif "OvE" in name:
            included_classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            positive_classes = [1, 3, 5, 7, 9]
        else:
            raise Exception("Dataset not supported")

        return __get_pu_mnist(
            included_classes=included_classes,
            positive_classes=positive_classes,
            case_control=case_control,
            scar=use_scar_labeling,
            label_frequency=label_frequency,
            data_dir=data_dir,
            val_size=val_size,
            test_size=test_size,
            synthetic_labels=synthetic_labels,
            optimize_intercept_only=optimize_intercept_only,
            max_sampling=max_sampling,
        )
    elif "CIFAR" in name:
        if "CarTruck" in name:
            included_classes = [1, 9]
            positive_classes = [1]
        elif "CarVsRest" in name:
            included_classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            positive_classes = [1]
        elif "MachineAnimal" in name:
            included_classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            positive_classes = [0, 1, 8, 9]
        else:
            raise Exception("Dataset not supported")

        return __get_pu_cifar10_precomputed(
            included_classes=included_classes,
            positive_classes=positive_classes,
            case_control=case_control,
            scar=use_scar_labeling,
            label_frequency=label_frequency,
            data_dir=data_dir,
            val_size=val_size,
            test_size=test_size,
            synthetic_labels=synthetic_labels,
            optimize_intercept_only=optimize_intercept_only,
            max_sampling=max_sampling,
        )
    elif "STL" in name:
        if "CarTruck" in name:
            included_classes = [1, 9]
            positive_classes = [1]
        elif "CarVsRest" in name:
            included_classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            positive_classes = [1]
        elif "MachineAnimal" in name:
            included_classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            positive_classes = [0, 1, 8, 9]
        else:
            raise Exception("Dataset not supported")

        return __get_pu_stl_precomputed(
            included_classes=included_classes,
            positive_classes=positive_classes,
            case_control=case_control,
            scar=use_scar_labeling,
            label_frequency=label_frequency,
            data_dir=data_dir,
            val_size=val_size,
            test_size=test_size,
            synthetic_labels=synthetic_labels,
            optimize_intercept_only=optimize_intercept_only,
            max_sampling=max_sampling,
        )
    elif "CDC-Diabetes" in name:
        if synthetic_labels:
            raise NotImplementedError()

        # https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators
        cdc_diabetes_health_indicators = fetch_ucirepo(id=891)

        # data (as pandas dataframes)
        X = cdc_diabetes_health_indicators.data.features.copy(deep=True)
        y = cdc_diabetes_health_indicators.data.targets

        scaler = MinMaxScaler()
        one_hot = OneHotEncoder()

        age_col = X["Age"]
        education_col = X["Education"]

        scaler_cols = ["BMI", "GenHlth", "MentHlth", "PhysHlth"]
        one_hot_cols = ["Age", "Education"]

        labeling_variables = pd.DataFrame(
            scaler.fit_transform(pd.concat([age_col, education_col], axis=1)),
            columns=["Age", "Education"],
        )

        X.loc[:, scaler_cols] = scaler.fit_transform(X.loc[:, scaler_cols])
        X_one_hot = pd.DataFrame(
            one_hot.fit_transform(X.loc[:, one_hot_cols]).toarray()
        )
        X = pd.concat([X.drop(columns=one_hot_cols), X_one_hot], axis=1)
        X = X.to_numpy()
        y = y.to_numpy()
        y = np.where(y.reshape(-1) == 1, 1, -1)

        age_score = 25 * (labeling_variables["Age"] + 1) ** 2  # 1 to 100
        edu_score = 100 * labeling_variables["Education"]
        labeling_score = age_score + edu_score

        positive_indices = np.where(y == 1)[0]
        positive_labeling_scores = labeling_score[positive_indices]
        positive_labeling_probas = positive_labeling_scores / np.sum(
            positive_labeling_scores
        )

        labeled_indices = np.random.choice(
            positive_indices,
            size=int(label_frequency * len(positive_indices)),
            replace=False,
            p=positive_labeling_probas,
        )
        o = np.where(np.isin(range(len(y)), labeled_indices), 1, 0)

        negative_indices = np.where(y == -1)[0]
        idx = np.concatenate(
            [
                positive_indices,
                np.random.choice(negative_indices, len(positive_indices)),
            ]
        )

        # idx = np.array(range(len(y)))
        # np.random.shuffle(idx)

        X = X[idx]
        y = y[idx]
        o = o[idx]

        X = torch.from_numpy(X)
        y = torch.from_numpy(y)
        o = torch.from_numpy(o)

        g, intercept = None, None

        return (
            *split_datasets(
                X, y, o, label_frequency, val_size, test_size, case_control=case_control
            ),
            g,
            intercept,
        )
    elif name.split(" - ")[0] in DATASET_NAMES:
        return get_small_dataset(
            name=name,
            device=device,
            case_control=case_control,
            scar=use_scar_labeling,
            label_frequency=label_frequency,
            val_size=val_size,
            test_size=test_size,
            synthetic_labels=synthetic_labels,
            optimize_intercept_only=optimize_intercept_only,
        )
    else:
        raise Exception("Dataset not supported")


def create_vae_pu_adapter(train, val, test, device="cuda"):
    x_train, y_train, s_train = train
    x_val, y_val, s_val = val
    x_test, y_test, s_test = test

    x_train, y_train, s_train = (
        torch.from_numpy(x_train).float(),
        torch.from_numpy(y_train).float(),
        torch.from_numpy(s_train).float(),
    )
    x_val, y_val, s_val = (
        torch.from_numpy(x_val).float(),
        torch.from_numpy(y_val).float(),
        torch.from_numpy(s_val).float(),
    )
    x_test, y_test, s_test = (
        torch.from_numpy(x_test).float(),
        torch.from_numpy(y_test).float(),
        torch.from_numpy(s_test).float(),
    )

    y_train = torch.where(y_train == 1, 1, -1)
    y_val = torch.where(y_val == 1, 1, -1)
    y_test = torch.where(y_test == 1, 1, -1)

    l_indices = torch.where(s_train == 1)
    u_indices = torch.where(s_train == 0)

    x_tr_l, y_tr_l = x_train[l_indices], y_train[l_indices]
    x_tr_u, y_tr_u = x_train[u_indices], y_train[u_indices]

    return (
        x_tr_l.to(device),
        y_tr_l.to(device),
        x_tr_u.to(device),
        y_tr_u.to(device),
        x_val.to(device),
        y_val.to(device),
        s_val.to(device),
        x_test.to(device),
        y_test.to(device),
        s_test.to(device),
    )


# %%
