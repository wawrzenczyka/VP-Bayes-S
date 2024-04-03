# %%
import json
import multiprocessing
import os
import threading
import time

import numpy as np
import tensorflow as tf
import torch

from config import config
from data_loading.vae_pu_dataloaders import (
    create_vae_pu_adapter,
    get_dataset,
    get_xs_data,
    get_xs_data_v2,
)
from external.LBE import predict_LBE, train_LBE
from external.nnPUlearning.api import nnPU
from external.sar_experiment import SAREMThreadedExperiment
from external.two_step import eval_2_step, train_2_step
from metrics_custom import calculate_metrics
from vae_pu_occ.vae_pu_occ_trainer import VaePuOccTrainer

label_frequencies = [0.9, 0.7, 0.5, 0.3, 0.1, 0.02]

start_idx = 0
num_experiments = 10
epoch_multiplier = 1

datasets = [
    "Synthetic (X, S) - logistic-interceptonly",
    "Synthetic (X, S) - logistic-interceptonly^10",
    "Synthetic (X, S) - 1-2-diagonal",
    "Synthetic (X, S) - SCAR",
    ### -------------------------------------------
    "MNIST 3v5",
    "CIFAR CarTruck",
    "STL MachineAnimal",
    "Gas Concentrations",
    "CDC-Diabetes",
    "MNIST OvE",
    "CIFAR MachineAnimal",
]

if "SCAR" in config["data"]:
    config["use_SCAR"] = True
else:
    config["use_SCAR"] = False

config["occ_methods"] = [
    "OddsRatio-PUprop-e100-lr1e-4-ES",
    "OddsRatio-e100-lr1e-4-ES",
    "SRuleOnly-e100-lr1e-4-ES",
]

config["use_original_paper_code"] = False
config["use_old_models"] = True

config["vae_pu_variant"] = None
config["nnPU_beta"], config["nnPU_gamma"] = None, None

if config["nnPU_beta"] is not None and config["nnPU_gamma"] is not None:
    config["vae_pu_variant"] = (
        f"beta_{config['nnPU_beta']:.0e}_gamma_{config['nnPU_gamma']:.0e}"
    )


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


for idx in range(start_idx, start_idx + num_experiments):
    for training_mode in [
        "S-Prophet",
        "Y-Prophet",
        "VAE-PU",
        "LBE",
    ]:
        config["training_mode"] = training_mode
        for dataset in datasets:
            config["data"] = dataset

            for base_label_frequency in label_frequencies:
                config["base_label_frequency"] = base_label_frequency

                np.random.seed(idx)
                torch.manual_seed(idx)
                tf.random.set_seed(idx)

                if config["device"] == "auto":
                    device = "cuda" if torch.cuda.is_available() else "cpu"

                if "Synthetic (X, S)" in config["data"]:
                    n_samples = 10_000
                    if "small" in config["data"]:
                        n_samples = 1_000
                    n_features = 20

                    synthetic_data_fun = get_xs_data
                    if "v2" in config["data"]:
                        synthetic_data_fun = get_xs_data_v2

                    if "SCAR" in config["data"]:
                        propensity_type = ""
                        scar = True
                    else:
                        scar = False
                        if "logistic" in config["data"]:
                            propensity_type = "logistic"

                        if "-inverse" in config["data"]:
                            propensity_type += "-inverse"
                        if "-separated" in config["data"]:
                            propensity_type += "-separated"
                        if "-interceptonly" in config["data"]:
                            propensity_type += "-interceptonly"
                        if "^10" in config["data"]:
                            propensity_type += "^10"
                        elif "1-2-diagonal" in config["data"]:
                            propensity_type = "1-2-diagonal"

                    (
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
                    ) = synthetic_data_fun(
                        base_label_frequency,
                        n_samples,
                        n_features,
                        scar=scar,
                        propensity_type=propensity_type,
                    )
                else:
                    (
                        train_samples,
                        val_samples,
                        test_samples,
                        label_frequency,
                        pi_p,
                        n_input,
                        g,
                        intercept,
                    ) = get_dataset(
                        config["data"],
                        device,
                        base_label_frequency,
                        use_scar_labeling=config["use_SCAR"],
                        synthetic_labels="Synthetic" in config["data"],
                        optimize_intercept_only="InterceptOnly" in config["data"],
                        max_sampling=" - proba sampling" not in config["data"],
                    )
                    y_proba_test, s_proba_test = None, None

                vae_pu_data = create_vae_pu_adapter(
                    train_samples, val_samples, test_samples, device
                )

                config["label_frequency"] = label_frequency
                config["pi_p"] = pi_p
                config["n_input"] = n_input

                config["pi_pl"] = label_frequency * pi_p
                config["pi_pu"] = pi_p - config["pi_pl"]
                config["pi_u"] = 1 - config["pi_pl"]

                config["g"] = g
                config["intercept"] = (
                    intercept.item() if type(intercept) == np.ndarray else intercept
                )
                config["c_error"] = np.abs(label_frequency - base_label_frequency)

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

                if (
                    "Synthetic (X, S)" in config["data"]
                    or "Gas Concentrations" in config["data"]
                ):
                    epoch_multiplier = 0.5

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
                elif (
                    config["data"] == "Synthetic (X, S)"
                    or "Gas Concentrations" in config["data"]
                ):
                    config["alpha_gen"] = 0.01
                    config["alpha_disc"] = 0.01
                    config["alpha_gen2"] = 0.3
                    config["alpha_disc2"] = 0.3

                    config["n_h_y"] = 10

                    config["n_hidden_pn"] = [200, 200]
                    config["n_hidden_vae_e"] = [100, 100]
                    config["n_hidden_vae_d"] = [100, 100]
                    config["n_hidden_disc"] = [20]

                    config["lr_pu"] = 3e-4
                    config["lr_pn"] = 3e-5
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
                    "result",
                    config["data"],
                    str(base_label_frequency),
                    "Exp" + str(idx),
                )

                if config["training_mode"] == "VAE-PU":
                    trainer = VaePuOccTrainer(
                        num_exp=idx,
                        model_config=config,
                        pretrain=True,
                        # ---
                        s_proba_test=s_proba_test,
                        y_proba_test=y_proba_test,
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
                    elif "Prophet" in config["training_mode"]:
                        if not "Synthetic (X, S)" in config["data"]:
                            continue

                        x, y, s = test_samples

                        metric_values = calculate_metrics(
                            y_proba_test,
                            y,
                            s_proba_test,
                            s,
                            use_s_rule=(config["training_mode"] == "S-Prophet"),
                            no_s_info_for_prediction=(
                                "-no S info" in config["training_mode"]
                            ),
                            method=config["training_mode"],
                        )

                        os.makedirs(method_dir, exist_ok=True)
                        with open(
                            os.path.join(method_dir, "metric_values.json"), "w"
                        ) as f:
                            json.dump(metric_values, f)
                    else:
                        if config["training_mode"] == "LBE":
                            log_prefix = f"{config['data']}, exp {idx}, c: {base_label_frequency} || "

                            lbe_training_start = time.perf_counter()
                            lbe = train_LBE(
                                train_samples,
                                val_samples,
                                verbose=True,
                                log_prefix=log_prefix,
                                training_lr=1e-3,
                                training_max_epochs=20,
                                iter_early_stopping_max_epochs=3,
                                M_step_early_stopping_max_epochs=10,
                            )
                            lbe_training_time = time.perf_counter() - lbe_training_start

                            for use_s_rule in [True, False]:
                                for no_s_info_for_prediction in [False, True]:
                                    if use_s_rule and no_s_info_for_prediction:
                                        continue
                                    method_full = (
                                        config["training_mode"]
                                        + (" +S rule" if use_s_rule else "")
                                        + (
                                            "-no S info"
                                            if no_s_info_for_prediction
                                            else ""
                                        )
                                    )

                                    y_proba, y, s_proba, s = predict_LBE(
                                        lbe, test_samples
                                    )

                                    metric_values = calculate_metrics(
                                        y_proba,
                                        y,
                                        s_proba,
                                        s,
                                        use_s_rule=use_s_rule,
                                        no_s_info_for_prediction=no_s_info_for_prediction,
                                        method=method_full,
                                        time=lbe_training_time,
                                    )

                                    print(
                                        f"{log_prefix}LBE accuracy: {100 * metric_values['Accuracy']:.2f}%"
                                    )
                                    print(
                                        f"{log_prefix}LBE F1-score: {100 * metric_values['F1 score']:.2f}%"
                                    )
                                    print(
                                        f"{log_prefix}LBE U-accuracy: {100 * metric_values['U-Accuracy']:.2f}%"
                                    )
                                    print(
                                        f"{log_prefix}LBE U-F1-score: {100 * metric_values['U-F1 score']:.2f}%"
                                    )

                                    method_dir = os.path.join(
                                        config["directory"], "external", method_full
                                    )

                                    os.makedirs(method_dir, exist_ok=True)
                                    with open(
                                        os.path.join(method_dir, "metric_values.json"),
                                        "w",
                                    ) as f:
                                        json.dump(metric_values, f)
                        else:
                            if config["training_mode"] == "2-step":
                                training_start = time.perf_counter()
                                clf = train_2_step(train_samples, config, idx)
                                training_time = time.perf_counter() - training_start

                                accuracy, precision, recall, f1 = eval_2_step(
                                    clf, test_samples
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

                                accuracy, precision, recall, f1 = clf.evaluate(
                                    test_samples
                                )

                            metric_values = {
                                "Method": config["training_mode"],
                                "Accuracy": accuracy,
                                "Precision": precision,
                                "Recall": recall,
                                "F1 score": f1,
                                "Time": training_time,
                            }

                            os.makedirs(method_dir, exist_ok=True)
                            with open(
                                os.path.join(method_dir, "metric_values.json"), "w"
                            ) as f:
                                json.dump(metric_values, f)

        for t in threads:
            t.join()
{}  # %%

# %%
