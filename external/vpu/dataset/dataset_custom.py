import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

custom_mean = (0.5,)
custom_std = (0.5,)


def normalize(x, mean=custom_mean, std=custom_std):
    x, mean, std = [np.array(a, np.float32) for a in (x, mean, std)]
    x -= mean
    x /= std
    return x


def transpose(x, source="NHWC", target="NCHW"):
    """
    N: batch size
    H: height
    W: weight
    C: channel
    """
    return x.transpose([source.index(d) for d in target])


def _to_4D(x):
    """
    :param x: For mnist, it is a tensor of shape (len, 28, 28)
    :return: a tensor of shape (len, 1, 28, 28)
    """
    return x.reshape(x.shape[0], 1, 28, 28)


def get_custom_data(x_l, x_u):
    x_l = torch.tensor(_to_4D(normalize(x_l)))
    x_u = torch.tensor(_to_4D(normalize(x_u)))

    x_l_train, x_l_val = val_split(x_l)
    x_u_train, x_u_val = val_split(x_u)

    train_labeled_dataset = TensorDataset(x_l_train, torch.ones(len(x_l_train)))
    train_unlabeled_dataset = TensorDataset(x_u_train, torch.zeros(len(x_u_train)))
    val_labeled_dataset = TensorDataset(x_l_val, torch.ones(len(x_l_val)))
    val_unlabeled_dataset = TensorDataset(x_u_val, torch.zeros(len(x_u_val)))

    return (
        train_labeled_dataset,
        train_unlabeled_dataset,
        val_labeled_dataset,
        val_unlabeled_dataset,
    )


def transform_test(x):
    x = torch.tensor(_to_4D(normalize(x)))

    test_dataset = TensorDataset(x)
    return test_dataset


def val_split(x):
    train_idx, val_idx = train_test_split(
        np.arange(len(x)), test_size=0.1, random_state=42, shuffle=True
    )

    return x[train_idx], x[val_idx]


def get_custom_loaders(x_l, x_u, batch_size=512):
    (
        train_labeled_dataset,
        train_unlabeled_dataset,
        val_labeled_dataset,
        val_unlabeled_dataset,
    ) = get_custom_data(x_l, x_u)

    p_loader = DataLoader(
        dataset=train_labeled_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    x_loader = DataLoader(
        dataset=train_unlabeled_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    val_p_loader = DataLoader(
        dataset=val_labeled_dataset, batch_size=batch_size, shuffle=False
    )
    val_x_loader = DataLoader(
        dataset=val_unlabeled_dataset, batch_size=batch_size, shuffle=False
    )
    return x_loader, p_loader, val_x_loader, val_p_loader


def get_custom_test_loader(x, batch_size=512):
    test_dataset = transform_test(x)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    return test_loader
