import math

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score

from .utils.checkpoint import *
from .utils.func import *


def run_vpu(config, loaders, NetworkPhi):
    """
    run VPU.

    :param config: arguments.
    :param loaders: loaders.
    :param NetworkPhi: class of the model.
    """

    lowest_val_var = math.inf  # lowest variational loss on validation set
    highest_test_acc = -1  # highest test accuracy on test set

    # set up loaders
    (p_loader, x_loader, val_p_loader, val_x_loader, test_loader) = loaders

    # set up model \Phi
    if config.dataset in ["cifar10", "fashionMNIST", "stl10"]:
        model_phi = NetworkPhi()
    elif config.dataset in ["pageblocks", "grid", "avila"]:
        input_size = len(p_loader.dataset[0][0])
        model_phi = NetworkPhi(input_size=input_size)
    if torch.cuda.is_available():
        model_phi = model_phi.cuda()

    # set up the optimizer
    lr_phi = config.learning_rate
    opt_phi = torch.optim.Adam(model_phi.parameters(), lr=lr_phi, betas=(0.5, 0.99))

    for epoch in range(config.epochs):
        # adjust the optimizer
        if epoch % 20 == 19:
            lr_phi /= 2
            opt_phi = torch.optim.Adam(
                model_phi.parameters(), lr=lr_phi, betas=(0.5, 0.99)
            )

        # train the model \Phi
        phi_loss, var_loss, reg_loss, phi_p_mean, phi_x_mean = train(
            config, model_phi, opt_phi, p_loader, x_loader
        )

        # evaluate the model \Phi
        val_var, test_acc, test_auc = evaluate(
            model_phi,
            x_loader,
            test_loader,
            val_p_loader,
            val_x_loader,
            epoch,
            phi_loss,
            var_loss,
            reg_loss,
        )

        # assessing performance of the current model and decide whether to save it
        is_val_var_lowest = val_var < lowest_val_var
        is_test_acc_highest = test_acc > highest_test_acc
        lowest_val_var = min(lowest_val_var, val_var)
        highest_test_acc = max(highest_test_acc, test_acc)
        if is_val_var_lowest:
            test_auc_of_best_val = test_auc
            test_acc_of_best_val = test_acc
            epoch_of_best_val = epoch
        save_checkpoint(
            {
                "epoch": epoch + 1,
                "state_dict": model_phi.state_dict(),
                "optimizer": opt_phi.state_dict(),
            },
            is_val_var_lowest,
            is_test_acc_highest,
            config=config,
        )

    # inform users model in which epoch is finally picked
    print(
        "Early stopping at {:}th epoch, test AUC : {:.4f}, test acc: {:.4f}".format(
            epoch_of_best_val, test_auc_of_best_val, test_acc_of_best_val
        )
    )


def train(
    config,
    model_phi,
    opt_phi,
    p_loader,
    x_loader,
    representation="DV",  # "DV" / "NJW" / "no-name"
    pi=None,
    use_extra_penalty=False,
    extra_penalty_config=None,
):
    """
    One epoch of the training of VPU.

    :param config: arguments.
    :param model_phi: current model \Phi.
    :param opt_phi: optimizer of \Phi.
    :param p_loader: loader for the labeled positive training data.
    :param x_loader: loader for training data (including positive and unlabeled)
    """

    # setup some utilities for analyzing performance
    phi_p_avg = AverageMeter()
    phi_x_avg = AverageMeter()
    phi_loss_avg = AverageMeter()
    var_loss_avg = AverageMeter()
    reg_avg = AverageMeter()

    # set the model to train mode
    model_phi.train()

    for batch_idx in range(config.val_iterations):
        try:
            data_x, _ = x_iter.next()
        except:
            x_iter = iter(x_loader)
            data_x, _ = x_iter.next()

        try:
            data_p, _ = p_iter.next()
        except:
            p_iter = iter(p_loader)
            data_p, _ = p_iter.next()

        if torch.cuda.is_available():
            data_p, data_x = data_p.cuda(), data_x.cuda()

        # calculate the variational loss
        data_all = torch.cat((data_p, data_x))

        # --- OLD CODE ---
        output_phi_all = model_phi(data_all)
        log_phi_all = output_phi_all[:, 1]
        # --- OLD CODE ---

        # --- NEW CODE ---
        if representation == "NJW":
            # * e / pi
            log_phi_all = log_phi_all + (1 - np.log(pi))
        elif representation == "no-name":
            # * 1 / pi
            log_phi_all = log_phi_all + np.log(1 / pi)
        # --- NEW CODE ---

        idx_p = slice(0, len(data_p))
        idx_x = slice(len(data_p), len(data_all))
        log_phi_x = log_phi_all[idx_x]
        log_phi_p = log_phi_all[idx_p]
        var_loss = (
            torch.logsumexp(log_phi_x, dim=0)
            - math.log(len(log_phi_x))
            - 1 * torch.mean(log_phi_p)
        )

        if representation != "new-variant":
            # perform Mixup and calculate the regularization
            target_x = log_phi_x.exp()
            target_p = torch.ones(len(data_p), dtype=torch.float32)
            target_p = target_p.cuda() if torch.cuda.is_available() else target_p
            rand_perm_p = torch.randperm(data_p.size(0))
            rand_perm_x = torch.randperm(data_x.size(0))
            data_p_perm, target_p_perm = data_p[rand_perm_p], target_p[rand_perm_p]

            m = torch.distributions.beta.Beta(config.mix_alpha, config.mix_alpha)
            lam = m.sample()

            data = lam * data_x + (1 - lam) * data_p_perm
            if torch.cuda.is_available():
                data = data.cuda()
            out_log_phi_all = model_phi(data)[:, 1]

            if "DistPU" in representation:
                log_phi_all_p_perm = log_phi_all[idx_p][rand_perm_p]
                log_phi_all_x_perm = log_phi_all[idx_x][rand_perm_x]
                if "KL" in representation:
                    import torch.nn.functional as F

                    reg_mix_log = (
                        lam * F.kl_div(log_phi_all_p_perm.exp(), out_log_phi_all.exp())
                    ) + (1 - lam) * F.kl_div(
                        log_phi_all_x_perm.exp(), out_log_phi_all.exp()
                    )
                elif "entropy" in representation:
                    reg_mix_log = -torch.mean(
                        log_phi_all_p_perm.exp() * out_log_phi_all
                        + (1 - log_phi_all_p_perm.exp())
                        * torch.log(1 - out_log_phi_all.exp())
                    )
            else:
                # normal mixup
                if representation == "DV":
                    target = lam * target_x + (1 - lam) * target_p_perm
                elif representation == "NJW":
                    target = (
                        lam * target_x + (1 - lam) * (np.exp(1) / pi) * target_p_perm
                    )
                elif representation == "no-name":
                    # target = lam * target_x + (1 - lam) * (1 / pi + 1) * target_p_perm
                    target = lam * target_x + (1 - lam) * (1 / pi) * target_p_perm
                else:
                    raise NotImplementedError("Unknown MixUp representation")

                if torch.cuda.is_available():
                    target = target.cuda()

                # --- NEW CODE ---
                if representation == "NJW":
                    # * e / pi
                    out_log_phi_all = out_log_phi_all + (1 - np.log(pi))
                elif representation == "no-name":
                    # * 1 / pi
                    out_log_phi_all = out_log_phi_all + np.log(1 / pi)
                # --- NEW CODE ---

                reg_mix_log = ((torch.log(target) - out_log_phi_all) ** 2).mean()
        else:
            # new variant
            phi_p = log_phi_p.exp()
            phi_x = log_phi_x.exp()
            reg_mix_log = (
                phi_p.mean() + config.mix_alpha * (phi_x.exp().mean() - 1) ** 2
            )

        # calculate gradients and update the network
        phi_loss = var_loss + config.lam * reg_mix_log

        if use_extra_penalty:
            out_phi_all = torch.exp(out_log_phi_all)
            if not extra_penalty_config["use_log"]:
                phi_loss += (
                    extra_penalty_config["lambda_pi"]
                    * (torch.mean(out_phi_all) - extra_penalty_config["pi"]) ** 2
                )
            else:
                phi_loss += (
                    extra_penalty_config["lambda_pi"]
                    * (
                        torch.log(torch.mean(out_phi_all))
                        - np.log(extra_penalty_config["pi"])
                    )
                    ** 2
                )

        opt_phi.zero_grad()
        phi_loss.backward()
        opt_phi.step()

        # update the utilities for analysis of the model
        reg_avg.update(reg_mix_log.item())
        phi_loss_avg.update(phi_loss.item())
        var_loss_avg.update(var_loss.item())
        phi_p, phi_x = log_phi_p.exp(), log_phi_x.exp()
        phi_p_avg.update(phi_p.mean().item(), len(phi_p))
        phi_x_avg.update(phi_x.mean().item(), len(phi_x))

    return phi_loss_avg.avg, var_loss_avg.avg, reg_avg.avg, phi_p_avg.avg, phi_x_avg.avg


def evaluate(
    model_phi,
    x_loader,
    test_loader,
    val_p_loader,
    val_x_loader,
    epoch,
    phi_loss,
    var_loss,
    reg_loss,
):
    """
    evaluate the performance on test set, and calculate the variational loss on validation set.

    :param model_phi: current model \Phi
    :param x_loader: loader for the whole training set (positive and unlabeled).
    :param test_loader: loader for the test set (fully labeled).
    :param val_p_loader: loader for positive data in the validation set.
    :param val_x_loader: loader for the whole validation set (including positive and unlabeled data).
    :param epoch: current epoch.
    :param phi_loss: VPU loss of the current epoch, which equals to var_loss + reg_loss.
    :param var_loss: variational loss of the training set.
    :param reg_loss: regularization loss of the training set.
    """

    # set the model to evaluation mode
    model_phi.eval()

    # calculate variational loss of the validation set consisting of PU data
    val_var = cal_val_var(model_phi, val_p_loader, val_x_loader)

    # max_phi is needed for normalization
    log_max_phi = -math.inf
    for idx, (data, _) in enumerate(x_loader):
        if torch.cuda.is_available():
            data = data.cuda()
        log_max_phi = max(log_max_phi, model_phi(data)[:, 1].max())

    # feed test set to the model and calculate accuracy and AUC
    with torch.no_grad():
        for idx, (data, target) in enumerate(test_loader):
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            log_phi = model_phi(data)[:, 1]
            log_phi -= log_max_phi
            if idx == 0:
                log_phi_all = log_phi
                target_all = target
            else:
                log_phi_all = torch.cat((log_phi_all, log_phi))
                target_all = torch.cat((target_all, target))
    pred_all = np.array((log_phi_all > math.log(0.5)).cpu().detach())
    log_phi_all = np.array(log_phi_all.cpu().detach())
    target_all = np.array(target_all.cpu().detach())
    test_acc = accuracy_score(target_all, pred_all)
    test_auc = roc_auc_score(target_all, log_phi_all)
    print(
        "Train Epoch: {}\t phi_loss: {:.4f}   var_loss: {:.4f}   reg_loss: {:.4f}   Test accuracy: {:.4f}   Val var loss: {:.4f}".format(
            epoch, phi_loss, var_loss, reg_loss, test_acc, val_var
        )
    )
    return val_var, test_acc, test_auc


def evaluate_val(
    model_phi, x_loader, val_p_loader, val_x_loader, epoch, phi_loss, var_loss, reg_loss
):
    """
    evaluate the performance on test set, and calculate the variational loss on validation set.

    :param model_phi: current model \Phi
    :param x_loader: loader for the whole training set (positive and unlabeled).
    :param val_p_loader: loader for positive data in the validation set.
    :param val_x_loader: loader for the whole validation set (including positive and unlabeled data).
    :param epoch: current epoch.
    :param phi_loss: VPU loss of the current epoch, which equals to var_loss + reg_loss.
    :param var_loss: variational loss of the training set.
    :param reg_loss: regularization loss of the training set.
    """

    # set the model to evaluation mode
    model_phi.eval()

    # calculate variational loss of the validation set consisting of PU data
    val_var = cal_val_var(model_phi, val_p_loader, val_x_loader)

    print(
        "Train Epoch: {}\t phi_loss: {:.4f}   var_loss: {:.4f}   reg_loss: {:.4f}   Val var loss: {:.4f}".format(
            epoch, phi_loss, var_loss, reg_loss, val_var
        )
    )
    return val_var


def cal_val_var(model_phi, val_p_loader, val_x_loader):
    """
    Calculate variational loss on the validation set, which consists of only positive and unlabeled data.

    :param model_phi: current \Phi model.
    :param val_p_loader: loader for positive data in the validation set.
    :param val_x_loader: loader for the whole validation set (including positive and unlabeled data).
    """

    # set the model to evaluation mode
    model_phi.eval()

    # feed the validation set to the model and calculate variational loss
    with torch.no_grad():
        for idx, (data_x, _) in enumerate(val_x_loader):
            if torch.cuda.is_available():
                data_x = data_x.cuda()
            output_phi_x_curr = model_phi(data_x)
            if idx == 0:
                output_phi_x = output_phi_x_curr
            else:
                output_phi_x = torch.cat((output_phi_x, output_phi_x_curr))
        for idx, (data_p, _) in enumerate(val_p_loader):
            if torch.cuda.is_available():
                data_p = data_p.cuda()
            output_phi_p_curr = model_phi(data_p)
            if idx == 0:
                output_phi_p = output_phi_p_curr
            else:
                output_phi_p = torch.cat((output_phi_p, output_phi_p_curr))
        log_phi_p = output_phi_p[:, 1]
        log_phi_x = output_phi_x[:, 1]
        var_loss = (
            torch.logsumexp(log_phi_x, dim=0)
            - math.log(len(log_phi_x))
            - torch.mean(log_phi_p)
        )
        return var_loss.item()
