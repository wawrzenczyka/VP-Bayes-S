# %%
import json
import multiprocessing
import os
import threading
import time

import numpy as np
import tensorflow as tf
import torch
from IPython.display import display

from config import config
from data_loading.vae_pu_dataloaders import create_vae_pu_adapter, get_dataset
from external.LBE import eval_LBE, train_LBE
from external.nnPUlearning.api import nnPU
from external.nnPUss.api import nnPUv2
from external.sar_experiment import SAREMThreadedExperiment
from external.two_step import eval_2_step, train_2_step
from vae_pu_occ.vae_pu_occ_trainer import VaePuOccTrainer

# label_frequencies = [0.7, 0.5, 0.3, 0.1, 0.02]
label_frequencies = [0.5, 0.02]
# label_frequencies = [0.02]
# label_frequencies = [0.7, 0.5, 0.3, 0.02]
# label_frequencies = [0.3, 0.7]
# label_frequencies = [0.02, 0.5]

start_idx = 0
num_experiments = 10
epoch_multiplier = 1


# config["occ_methods"] = ["OC-SVM", "IsolationForest", "ECODv2", "A^3"]
# config["occ_methods"] = ["MixupPU", "EM-PU", "MixupPU+concat"]
# config["occ_methods"] = ["MixupPU+concat"]
config["occ_methods"] = [
    # "MixupPU",
    # "MixupPU-NJW",
    # "MixupPU-NJW-no-norm",
    # "MixupPU-no-name",
    # "MixupPU-no-name-no-norm",
    # "MixupPU+extra-loss-3",
    # "MixupPU+extra-loss-1",
    # "MixupPU+extra-loss-0.3",
    # "MixupPU+extra-loss-0.1",
    # "MixupPU+extra-loss-0.03",
    # "MixupPU+extra-loss-0.003",
    # "MixupPU+extra-loss-log-3",
    # "MixupPU+extra-loss-log-1",
    # "MixupPU+extra-loss-log-0.3",
    # "MixupPU+extra-loss-log-0.1",
    # "MixupPU+extra-loss-log-0.03",
    # "MixupPU+extra-loss-log-0.003",
]
# config["occ_methods"] = ["EM-PU"]

config["use_original_paper_code"] = False
# config['use_original_paper_code'] = True
config["use_old_models"] = True
# config['use_old_models'] = False

# config["training_mode"] = "VAE-PU"
# config['training_mode'] = 'SAR-EM'
# config['training_mode'] = 'LBE'
# config["training_mode"] = "2-step"

# config["training_mode"] = "nnPU"
# config["training_mode"] = "nnPUss"
# config["training_mode"] = "uPU"

# config["training_mode"] = "MixupPU"
# # config["training_mode"] = "MixupPU-NJW"
# # config["training_mode"] = "MixupPU-no-name"

# # config["training_mode"] = "RawOCC-IsolationForest"
# # config["training_mode"] = "RawOCC-OC-SVM"
# # config["training_mode"] = "RawOCC-A^3"
# # config["training_mode"] = "RawOCC-ECODv2"


config["use_case_control_data"] = True

config["nnPU_beta"], config["nnPU_gamma"] = None, None
# config["nnPU_beta"], config["nnPU_gamma"] = 0, 1
# config["nnPU_beta"], config["nnPU_gamma"] = 1e-3, 1
# config["nnPU_beta"], config["nnPU_gamma"] = 1e-2, 1
# config["nnPU_beta"], config["nnPU_gamma"] = 1e-4, 1
# config["nnPU_beta"], config["nnPU_gamma"] = 5e-3, 1
# config["nnPU_beta"], config["nnPU_gamma"] = 5e-2, 1
# config["nnPU_beta"], config["nnPU_gamma"] = 5e-4, 1
# config["nnPU_beta"], config["nnPU_gamma"] = 1e-3, 0.5
# config["nnPU_beta"], config["nnPU_gamma"] = 1e-2, 0.5
# config["nnPU_beta"], config["nnPU_gamma"] = 1e-4, 0.5
# config["nnPU_beta"], config["nnPU_gamma"] = 5e-3, 0.5
# config["nnPU_beta"], config["nnPU_gamma"] = 5e-2, 0.5
# config["nnPU_beta"], config["nnPU_gamma"] = 5e-4, 0.5

if config["nnPU_beta"] is not None and config["nnPU_gamma"] is not None:
    config[
        "vae_pu_variant"
    ] = f"beta_{config['nnPU_beta']:.0e}_gamma_{config['nnPU_gamma']:.0e}"


config["train_occ"] = True
config["occ_num_epoch"] = round(100 * epoch_multiplier)

config["early_stopping"] = True
config["early_stopping_epochs"] = 10

if config["use_original_paper_code"]:
    config["mode"] = "near_o"
else:
    config["mode"] = "near_y"

config["device"] = "auto"

# used by SAR-EM
n_threads = multiprocessing.cpu_count()
sem = threading.Semaphore(n_threads)
threads = []

for dataset in [
    # "MNIST 3v5",
    # "MNIST OvE",
    # "MNIST 3v5 SCAR",
    "MNIST OvE SCAR",
]:
    config["data"] = dataset
    if "SCAR" in config["data"]:
        config["use_SCAR"] = True
    else:
        config["use_SCAR"] = False

    for training_mode in [
        # "MixupPU",
        # "MixupPU-NJW",
        # "MixupPU-NJW-no-norm",
        # "MixupPU-no-name",
        # "MixupPU-no-name-no-norm",
        # "MixupPU-new-variant",
        # "MixupPU-DistPU-penalty-KL",
        # "MixupPU-DistPU-penalty-entropy",
        # "RawOCC-IsolationForest",
        # "RawOCC-OC-SVM",
        # "RawOCC-A^3",
        # "RawOCC-ECODv2",
        # "RawOCC-IsolationForest-LR",
        # "RawOCC-OC-SVM-LR",
        # "RawOCC-A^3-LR",
        # "RawOCC-ECODv2-LR",
        # "RawOCC-IsolationForest-FOR-CTL",
        # "RawOCC-OC-SVM-FOR-CTL",
        # "RawOCC-A^3-FOR-CTL",
        # "RawOCC-ECODv2-FOR-CTL",
        # "RawOCC-IsolationForest-LR-FOR-CTL",
        # "RawOCC-OC-SVM-LR-FOR-CTL",
        # "RawOCC-A^3-LR-FOR-CTL",
        # "RawOCC-ECODv2-LR-FOR-CTL",
        # "nnPU",
        # "nnPUv2",
        "nnPUss",
        # "nnPUssv2",
        # "uPU",
    ]:
        config["training_mode"] = training_mode

        for idx in range(start_idx, start_idx + num_experiments):
            for base_label_frequency in label_frequencies:
                config["base_label_frequency"] = base_label_frequency

                np.random.seed(idx)
                torch.manual_seed(idx)
                tf.random.set_seed(idx)

                if config["device"] == "auto":
                    device = "cuda" if torch.cuda.is_available() else "cpu"

                (
                    train_samples,
                    val_samples,
                    test_samples,
                    label_frequency,
                    pi_p,
                    n_input,
                ) = get_dataset(
                    config["data"],
                    device,
                    base_label_frequency,
                    case_control=config["use_case_control_data"],
                    use_scar_labeling=config["use_SCAR"],
                )
                vae_pu_data = create_vae_pu_adapter(
                    train_samples, val_samples, test_samples, device
                )

                config["label_frequency"] = label_frequency
                config["pi_p"] = pi_p
                config["n_input"] = n_input

                config["pi_pl"] = label_frequency * pi_p
                config["pi_pu"] = pi_p - config["pi_pl"]
                config["pi_u"] = 1 - config["pi_pl"]

                batch_size = 1000
                pl_batch_size = int(np.ceil(config["pi_pl"] * batch_size))
                u_batch_size = batch_size - pl_batch_size
                config["batch_size_l"], config["batch_size_u"] = (
                    pl_batch_size,
                    u_batch_size,
                )
                config["batch_size_l_pn"], config["batch_size_u_pn"] = (
                    pl_batch_size,
                    u_batch_size,
                )

                config["n_h_y"] = 10
                config["n_h_o"] = 2
                config["lr_pu"] = 3e-4
                config["lr_pn"] = 1e-5

                config["num_epoch_pre"] = round(100 * epoch_multiplier)
                config["num_epoch_step1"] = round(400 * epoch_multiplier)
                config["num_epoch_step_pn1"] = round(500 * epoch_multiplier)
                config["num_epoch_step_pn2"] = round(600 * epoch_multiplier)
                config["num_epoch_step2"] = round(500 * epoch_multiplier)
                config["num_epoch_step3"] = round(700 * epoch_multiplier)
                config["num_epoch"] = round(800 * epoch_multiplier)

                config["n_hidden_cl"] = []
                config["n_hidden_pn"] = [300, 300, 300, 300]

                if config["data"] == "MNIST OvE":
                    config["alpha_gen"] = 0.1
                    config["alpha_disc"] = 0.1
                    config["alpha_gen2"] = 3
                    config["alpha_disc2"] = 3
                elif ("CIFAR" in config["data"] or "STL" in config["data"]) and config[
                    "use_SCAR"
                ]:
                    config["alpha_gen"] = 3
                    config["alpha_disc"] = 3
                    config["alpha_gen2"] = 1
                    config["alpha_disc2"] = 1
                    ### What is it?
                    config["alpha_test"] = 1.0
                elif (
                    "CIFAR" in config["data"] or "STL" in config["data"]
                ) and not config["use_SCAR"]:
                    config["alpha_gen"] = 0.3
                    config["alpha_disc"] = 0.3
                    config["alpha_gen2"] = 1
                    config["alpha_disc2"] = 1
                    ### What is it?
                    config["alpha_test"] = 1.0
                else:
                    config["alpha_gen"] = 1
                    config["alpha_disc"] = 1
                    config["alpha_gen2"] = 10
                    config["alpha_disc2"] = 10

                config["device"] = device
                config["directory"] = os.path.join(
                    "result-cc",
                    config["data"],
                    str(base_label_frequency),
                    "Exp" + str(idx),
                )

                if config["training_mode"] == "VAE-PU":
                    trainer = VaePuOccTrainer(
                        num_exp=idx, model_config=config, pretrain=True
                    )
                    trainer.train(vae_pu_data)
                else:
                    np.random.seed(idx)
                    torch.manual_seed(idx)
                    tf.random.set_seed(idx)
                    method_dir = os.path.join(
                        config["directory"], "external", config["training_mode"]
                    )

                    if config["training_mode"] == "SAR-EM":
                        exp_thread = SAREMThreadedExperiment(
                            train_samples,
                            test_samples,
                            idx,
                            base_label_frequency,
                            config,
                            method_dir,
                            sem,
                        )
                        exp_thread.start()
                        threads.append(exp_thread)
                    else:
                        if config["training_mode"] == "LBE":
                            log_prefix = f"Exp {idx}, c: {base_label_frequency} || "

                            lbe_training_start = time.perf_counter()
                            lbe = train_LBE(
                                train_samples,
                                val_samples,
                                verbose=True,
                                log_prefix=log_prefix,
                            )
                            lbe_training_time = time.perf_counter() - lbe_training_start

                            accuracy, precision, recall, f1 = eval_LBE(
                                lbe, test_samples, verbose=True, log_prefix=log_prefix
                            )
                        elif config["training_mode"] == "2-step":
                            training_start = time.perf_counter()
                            clf = train_2_step(train_samples, config, idx)
                            training_time = time.perf_counter() - training_start

                            accuracy, precision, recall, f1 = eval_2_step(
                                clf, test_samples
                            )
                        elif "MixupPU" in config["training_mode"]:
                            x, y, s = train_samples
                            x_l = x[s == 1]
                            x_u = x[s == 0]

                            use_extra_penalty = False
                            extra_penalty_config = {
                                "lambda_pi": 0.03,
                                "pi": config["pi_p"],
                                "use_log": False,
                            }

                            representation = "DV"
                            if "NJW" in config["training_mode"]:
                                representation = "NJW"
                            elif "no-name" in config["training_mode"]:
                                representation = "no-name"
                            elif "new-variant" in config["training_mode"]:
                                representation = "new-variant"
                            elif "DistPU-penalty-KL" in config["training_mode"]:
                                representation = "DistPU-KL"
                            elif "DistPU-penalty-entropy" in config["training_mode"]:
                                representation = "DistPU-entropy"

                            normalize_phi = True
                            if "no-norm" in config["training_mode"]:
                                normalize_phi = False

                            from external.vpu.api import VPU

                            mixup_model = VPU(
                                representation=representation,
                                pi=config["pi_p"],
                                normalize_phi=normalize_phi,
                                use_extra_penalty=use_extra_penalty,
                                extra_penalty_config=extra_penalty_config,
                                batch_size=128 if base_label_frequency != 0.02 else 16,
                            )
                            training_start = time.perf_counter()
                            mixup_model.train(
                                x_l=torch.from_numpy(x_l),
                                x_u=torch.from_numpy(x_u),
                            )
                            training_time = time.perf_counter() - training_start

                            x_test, y_test, _ = test_samples
                            y_pred = mixup_model.predict(torch.from_numpy(x_test))
                            y_pred = np.where(y_pred == 1, 1, -1)

                            from sklearn import metrics

                            accuracy, precision, recall, f1 = (
                                metrics.accuracy_score(y_test, y_pred),
                                metrics.precision_score(y_test, y_pred),
                                metrics.recall_score(y_test, y_pred),
                                metrics.f1_score(y_test, y_pred),
                            )
                        elif "RawOCC" in config["training_mode"]:
                            x, y, s = train_samples
                            x_l = x[s == 1]
                            x_u = x[s == 0]

                            from sklearn.ensemble import IsolationForest
                            from sklearn.svm import OneClassSVM

                            from external.A3_adapter import A3Adapter
                            from external.cccpv.methods import ConformalPvalues
                            from external.ecod_v2 import ECODv2
                            from external.occ_cutoffs import (
                                EmpiricalCutoff,
                                FORControlCutoff,
                                MultisplitCutoff,
                            )
                            from external.pyod_wrapper import PyODWrapper

                            contamination = 0.001
                            contamination = min(max(contamination, 0.004), 0.1)
                            if "OC-SVM" in config["training_mode"]:
                                construct_clf = lambda: OneClassSVM(
                                    nu=contamination, kernel="rbf", gamma=0.1
                                )
                            elif "IsolationForest" in config["training_mode"]:
                                construct_clf = lambda: IsolationForest(
                                    random_state=idx, contamination=contamination
                                )
                            elif "A^3" in config["training_mode"]:
                                construct_clf = lambda: A3Adapter(
                                    target_epochs=10, a3_epochs=10
                                )
                            elif "ECODv2" in config["training_mode"]:
                                construct_clf = lambda: PyODWrapper(
                                    ECODv2(contamination=contamination, n_jobs=-2)
                                )

                            training_start = time.perf_counter()

                            if "FOR-CTL" in config["training_mode"]:
                                base_cutoff = MultisplitCutoff(
                                    construct_clf,
                                    alpha=0.05,
                                    resampling_repeats=20,
                                    n_jobs=-2
                                    if "OC-SVM" in config["training_mode"]
                                    else 1,
                                )
                                base_cutoff.fit(x_l)
                                # backup_clf = EmpiricalCutoff(construct_clf)
                                # backup_clf.fit(x_l)
                            else:
                                cc = ConformalPvalues(
                                    x_l,
                                    construct_clf(),
                                    calib_size=0.5,
                                    random_state=idx,
                                )

                            if "LR" in config["training_mode"]:
                                pu_to_u_ratio = config["pi_pu"] / config["pi_u"]

                                if "FOR-CTL" in config["training_mode"]:
                                    clf = FORControlCutoff(
                                        base_cutoff, base_cutoff.alpha, pi=pu_to_u_ratio
                                    )
                                    pvals, y_pred = clf.fit_apply(x_u)
                                    is_pu = torch.from_numpy(y_pred == 1)
                                else:
                                    # train downstream Logistic Regression
                                    pvals_one_class = cc.predict(
                                        x_u, delta=0.05, simes_kden=2
                                    )
                                    pvals = pvals_one_class["Marginal"]
                                    pvals = torch.from_numpy(pvals)

                                    # Order approach
                                    sorted_indices = torch.argsort(
                                        pvals, descending=True
                                    )
                                    n_p_samples = round(pu_to_u_ratio * len(x_u))
                                    pu_indices = sorted_indices[:n_p_samples]
                                    is_pu = np.where(
                                        np.isin(np.arange(len(pvals)), pu_indices),
                                        True,
                                        False,
                                    )

                                x_p = np.concatenate([x_l, x_u[is_pu]])
                                x_n = x_u[~is_pu]

                                X_train_lr = np.concatenate(
                                    [
                                        x_p,
                                        x_n,
                                    ]
                                )
                                y_train_lr = np.concatenate(
                                    [
                                        np.ones(len(x_p)),
                                        np.zeros(len(x_n)),
                                    ]
                                )

                                from sklearn.linear_model import LogisticRegression

                                lr = LogisticRegression()
                                lr.fit(X_train_lr, y_train_lr)
                                training_time = time.perf_counter() - training_start

                                x_test, y_test, _ = test_samples
                                y_pred = np.where(
                                    lr.predict(x_test) == 1,
                                    1,
                                    -1,
                                )
                            else:
                                training_time = time.perf_counter() - training_start

                                # predict using OCC
                                x_test, y_test, _ = test_samples

                                if "FOR-CTL" in config["training_mode"]:
                                    clf = FORControlCutoff(
                                        base_cutoff,
                                        base_cutoff.alpha,
                                        pi=config["pi_p"],
                                    )
                                    pvals, y_pred = clf.fit_apply(x_test)
                                    y_pred = np.where(y_pred == 1, 1, -1)
                                else:
                                    pvals_one_class = cc.predict(
                                        x_test, delta=0.05, simes_kden=2
                                    )
                                    pvals = pvals_one_class["Marginal"]
                                    pvals = torch.from_numpy(pvals)

                                    # Order approach
                                    sorted_indices = torch.argsort(
                                        pvals, descending=True
                                    )
                                    n_p_samples = round(config["pi_p"] * len(y_test))
                                    pu_indices = sorted_indices[:n_p_samples]
                                    y_pred = np.where(
                                        np.isin(np.arange(len(pvals)), pu_indices),
                                        1,
                                        -1,
                                    )

                            from sklearn import metrics

                            accuracy, precision, recall, f1 = (
                                metrics.accuracy_score(y_test, y_pred),
                                metrics.precision_score(y_test, y_pred),
                                metrics.recall_score(y_test, y_pred),
                                metrics.f1_score(y_test, y_pred),
                            )
                        elif config["training_mode"] in ["nnPUss", "nnPU", "uPU"]:
                            x, y, s = train_samples
                            x = x.reshape(-1, 1, 28, 28)
                            x = (x + 1) / 2
                            y = y.astype(np.int64)
                            s = s.astype(np.int64)
                            train_samples = x, y, s

                            training_start = time.perf_counter()
                            clf = nnPU(model_name=config["training_mode"])
                            clf.train(train_samples, config["pi_p"])
                            training_time = time.perf_counter() - training_start

                            x, y, s = test_samples
                            x = x.reshape(-1, 1, 28, 28)
                            x = (x + 1) / 2
                            y = y.astype(np.int64)
                            s = s.astype(np.int64)
                            test_samples = x, y, s

                            accuracy, precision, recall, f1 = clf.evaluate(test_samples)
                        elif config["training_mode"] in ["nnPUssv2", "nnPUv2"]:
                            x, y, s = train_samples
                            x = x.reshape(-1, 1, 28, 28)
                            x = (x + 1) / 2
                            y = y.astype(np.int64)
                            s = s.astype(np.int64)
                            train_samples = x, y, s

                            x, y, s = test_samples
                            x = x.reshape(-1, 1, 28, 28)
                            x = (x + 1) / 2
                            y = y.astype(np.int64)
                            s = s.astype(np.int64)
                            test_samples = x, y, s

                            training_start = time.perf_counter()
                            clf = nnPUv2(model_name=config["training_mode"])
                            clf.train(
                                train_samples,
                                val_samples=test_samples,
                                pi=config["pi_p"],
                                batch_size=512,
                                lr=1e-5,
                            )
                            training_time = time.perf_counter() - training_start

                            accuracy, precision, recall, f1 = clf.evaluate(test_samples)

                        metric_values = {
                            "Method": config["training_mode"],
                            "Accuracy": accuracy,
                            "Precision": precision,
                            "Recall": recall,
                            "F1 score": f1,
                            "Time": training_time,
                        }
                        display(metric_values)

                        # os.makedirs(method_dir, exist_ok=True)
                        # with open(
                        #     os.path.join(method_dir, "metric_values.json"), "w"
                        # ) as f:
                        #     json.dump(metric_values, f)

        for t in threads:
            t.join()

# %%
