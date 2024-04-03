import copy
import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn import metrics
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from data_loading.utils import InfiniteDataLoader
from metrics_custom import calculate_metrics

from .model import (
    VAEdecoder,
    VAEencoder,
    classifier_o,
    classifier_pn,
    discriminator,
    myPU,
)


class VaePuTrainer:
    def __init__(self, num_exp, model_config, pretrain=True):
        self.num_exp = num_exp
        self.config = model_config
        self.use_original_paper_code = self.config["use_original_paper_code"]
        self.device = self.config["device"]
        self.pretrain = pretrain

    def train(self, vae_pu_data):
        self._prepare_dataloaders(vae_pu_data)
        self._prepare_model()
        self._prepare_metrics()

        trained_vae_pu_exists = os.path.exists(
            os.path.join(self.config["directory"], "model_pre_occ.pt")
        )
        if (
            self.config["use_old_models"]
            and not self.use_original_paper_code
            and self.config["vae_pu_variant"] is None
            and trained_vae_pu_exists
        ):
            self.model = self._load_trained_vae_pu()
        else:
            self.baseline_training_start = time.perf_counter()
            self._prepare_log_files()

            if self.pretrain:
                self._pretrain_autoencoder()

            self.model.findPrior(self.x_pl_full, self.x_u_full)

            self.model_post_vae_training = None
            for epoch in range(self.config["num_epoch"]):
                if not self.use_original_paper_code:
                    self.log = open(
                        os.path.join(self.config["directory"], "log.txt"), "a"
                    )
                    self.log2 = open(
                        os.path.join(self.config["directory"], "log_PN.txt"), "a"
                    )

                start_time = time.time()
                epoch_losses = {
                    "ELBO": [],
                    "Adversarial generation": [],
                    "Discriminator": [],
                    "Label": [],
                    "Target classifier": [],
                }

                # training
                for x_pl, x_u in zip(self.DL_pl, self.DL_u):
                    x_pl, x_u = x_pl[0], x_u[0]

                    if epoch < self.config["num_epoch_step1"]:
                        # train autoencoder only, no label loss
                        self._train_autoencoder(epoch, x_pl, x_u, epoch_losses)
                    # num_epoch_step2 == num_epoch_step_pn1
                    elif epoch < self.config["num_epoch_step2"]:
                        # train target classifier only
                        self._train_target_classifier(epoch, x_pl, x_u, epoch_losses)
                    elif epoch < self.config["num_epoch_step_pn2"]:
                        # train autoencoder only, with label loss
                        self._train_autoencoder(epoch, x_pl, x_u, epoch_losses)
                    elif epoch < self.config["num_epoch_step3"]:
                        # train both
                        self._train_autoencoder(epoch, x_pl, x_u, epoch_losses)
                        self._train_target_classifier(epoch, x_pl, x_u, epoch_losses)
                    else:
                        # finish by training only target classifier
                        self._train_target_classifier(epoch, x_pl, x_u, epoch_losses)

                # metrics and results
                if epoch < self.config["num_epoch_step1"]:
                    # train autoencoder only, no label loss
                    if (epoch + 1) % 100 == 0:
                        self._check_discriminator_and_classifier(epoch, x_pl, x_u)
                    self._print_autoencoder_log(epoch, epoch_losses)
                    self._calculate_VAE_metrics()
                    self.timesAutoencoder.append(time.time() - start_time)
                elif epoch < self.config["num_epoch_step2"]:
                    # train target classifier only
                    if (epoch + 1) % 100 == 0:
                        self._check_discriminator_and_classifier(epoch, x_pl, x_u)
                    self._calculate_target_classifier_metrics(epoch, epoch_losses)
                    self.timesTargetClassifier.append(time.time() - start_time)
                elif epoch < self.config["num_epoch_step_pn2"]:
                    # train autoencoder only, with label loss
                    if (epoch + 1) % 100 == 0:
                        self._check_discriminator_and_classifier(epoch, x_pl, x_u)
                    self._print_autoencoder_log(epoch, epoch_losses)
                    self._calculate_VAE_metrics()
                    self.timesAutoencoder.append(time.time() - start_time)
                elif epoch < self.config["num_epoch_step3"]:
                    # train both
                    if (epoch + 1) % 100 == 0:
                        self._check_discriminator_and_classifier(epoch, x_pl, x_u)
                    self._print_autoencoder_log(epoch, epoch_losses)
                    self._calculate_target_classifier_metrics(epoch, epoch_losses)
                    self.timesAutoencoder.append(time.time() - start_time)
                    self.timesTargetClassifier.append(time.time() - start_time)
                else:
                    # finish by training only target classifier
                    self._calculate_target_classifier_metrics(epoch, epoch_losses)
                    self.timesTargetClassifier.append(time.time() - start_time)

                if not self.use_original_paper_code:
                    self.log.close()
                    self.log2.close()

                print(
                    f'Exp: {self.num_exp} / c = {self.config["base_label_frequency"]:.2f} / Epoch: {epoch + 1:4} |||| Remaining time (baseline): {(self.config["num_epoch"] - epoch) * (time.time() - start_time):.2f} sec'
                )
            self.baseline_training_time = (
                time.perf_counter() - self.baseline_training_start
            )

            y_probas = []
            y_trues = []
            s_trues = []
            for x, y, s, idx in self.DL_test_s:
                y_proba = self.model.model_pn.classify(x, sigmoid=True).reshape(-1)
                y_true = torch.where(y == 1, 1, 0)
                y_probas.append(y_proba)
                y_trues.append(y_true)
                s_trues.append(s)
            y_proba = torch.cat(y_probas).detach().cpu().numpy()
            y_true = torch.cat(y_trues).detach().cpu().numpy()
            s_true = torch.cat(s_trues).detach().cpu().numpy()

            method = "No OCC"
            if self.use_original_paper_code:
                method = "Baseline (orig)"
            elif self.config["vae_pu_variant"] is not None:
                method = f"VAE-PU-{self.config['vae_pu_variant']}"

            metric_values = calculate_metrics(
                y_proba,
                y_true,
                np.empty_like(s_true),
                s_true,
                use_s_rule=False,
                method=method,
                time=self.baseline_training_time,
            )
            self._save_final_vae_pu_metric_values(metric_values)
        return self.model

    def _prepare_metrics(self):
        self.elbos = []
        self.advGenerationLosses = []
        self.discLosses = []
        self.labelLosses = []
        self.targetClassifierLosses = []
        self.valAccuracies = []
        self.valLosses = []

        self.timesAutoencoder = []
        self.timesTargetClassifier = []

    def _save_final_vae_pu_metric_values(self, metric_values):
        metrics_path = os.path.join(self.config["directory"], "metric_values.json")
        if self.use_original_paper_code:
            metrics_path = os.path.join(
                self.config["directory"], "metric_values_orig.json"
            )
        elif self.config["vae_pu_variant"] is not None:
            metrics_path = os.path.join(
                self.config["directory"],
                "variants",
                self.config["vae_pu_variant"],
                "metric_values.json",
            )
            os.makedirs(os.path.dirname(metrics_path), exist_ok=True)

        with open(metrics_path, "w") as f:
            json.dump(metric_values, f)

        if not self.use_original_paper_code:
            with open(
                os.path.join(self.config["directory"], "settings.json"), "w"
            ) as f:
                json.dump(
                    {
                        "Label frequency": self.config["label_frequency"].item(),
                        "Pi": self.config["pi_p"].item(),
                        "True storey pi": (
                            self.config["pi_pu"] / self.config["pi_u"]
                        ).item(),
                        "g": (
                            self.config["g"].tolist()
                            if type(self.config["g"]) == np.ndarray
                            else self.config["g"]
                        ),
                        "intercept": self.config["intercept"],
                        "c_error": self.config["c_error"],
                    },
                    f,
                )

            self._plotLoss(
                self.targetClassifierLosses,
                os.path.join(self.config["directory"], "loss_PN.png"),
            )
            self._plotLoss(
                self.valAccuracies,
                os.path.join(self.config["directory"], "val_accuracy.png"),
            )

            torch.save(
                self.model, os.path.join(self.config["directory"], "model_pre_occ.pt")
            )

            log2 = open(os.path.join(self.config["directory"], "log_PN.txt"), "a")
            acc, precision, recall, f1_score = self.model.accuracy(self.DL_test)

            if self.config["train_occ"]:
                log2.write(
                    "final test pre-occ: acc: {0:.4f}, precision: {1:.4f}, recall: {2:.4f}, f1-score: {3:.4f}".format(
                        acc, precision, recall, f1_score
                    )
                    + "\n"
                )
                print(
                    "final test pre-occ: acc: {0:.4f}, precision: {1:.4f}, recall: {2:.4f}, f1-score: {3:.4f}".format(
                        acc, precision, recall, f1_score
                    )
                )
            else:
                log2.write(
                    "final test : acc: {0:.4f}, precision: {1:.4f}, recall: {2:.4f}, f1-score: {3:.4f}".format(
                        acc, precision, recall, f1_score
                    )
                )
                print(
                    "final test : acc: {0:.4f}, precision: {1:.4f}, recall: {2:.4f}, f1-score: {3:.4f}".format(
                        acc, precision, recall, f1_score
                    )
                )
            log2.close()

    def _calculate_target_classifier_metrics(self, epoch, losses):
        targetLoss = np.mean(losses["Target classifier"])

        if not self.use_original_paper_code:
            self.log2.write("epoch: {}, loss: {}".format(epoch + 1, targetLoss) + "\n")
        print("epoch: {}, loss: {}".format(epoch + 1, targetLoss))

        val_acc, val_pr, val_re, val_f1 = self.model.accuracy(self.DL_val)
        self.valAccuracies.append(val_acc)
        if not self.use_original_paper_code:
            self.log2.write(
                "...val: acc: {0:.4f}, precision: {1:.4f}, recall: {2:.4f}, f1: {3:.4f}".format(
                    val_acc, val_pr, val_re, val_f1
                )
                + "\n"
            )
        print(
            "...val: acc: {0:.4f}, precision: {1:.4f}, recall: {2:.4f}, f1: {3:.4f}".format(
                val_acc, val_pr, val_re, val_f1
            )
        )

        val_loss = self.model.loss_val(self.x_val[:20], self.x_val[20:])
        self.valLosses.append(val_loss)
        print(val_loss)

        self.targetClassifierLosses.append(targetLoss)
        print(
            f'Exp: {self.num_exp} / c = {self.config["base_label_frequency"]:.2f} / Epoch: {epoch + 1:4} |||| PN loss: {targetLoss}'
        )

    def _calculate_VAE_metrics(self):
        val_acc, val_pr, val_re, val_f1 = self.model.accuracy(self.DL_val, use_vae=True)
        if not self.use_original_paper_code:
            self.log.write(
                "...(VAE) val: acc: {0:.4f}, precision: {1:.4f}, recall: {2:.4f}, f1: {3:.4f}".format(
                    val_acc, val_pr, val_re, val_f1
                )
                + "\n"
            )
        print(
            "...(VAE) val: acc: {0:.4f}, precision: {1:.4f}, recall: {2:.4f}, f1: {3:.4f}".format(
                val_acc, val_pr, val_re, val_f1
            )
        )

    def _print_autoencoder_log(self, epoch, losses):
        elbo, advGenLoss, discLoss, labelLoss = (
            np.mean(losses["ELBO"]),
            np.mean(losses["Adversarial generation"]),
            np.mean(losses["Discriminator"]),
            np.mean(losses["Label"]),
        )

        self.elbos.append(elbo)
        self.advGenerationLosses.append(advGenLoss)
        self.discLosses.append(discLoss)
        self.labelLosses.append(labelLoss)

        print(
            f'Exp: {self.num_exp} / c = {self.config["base_label_frequency"]:.2f} / Epoch: {epoch + 1:4} |||| ELBO loss: {elbo:.4f}, AdvGen loss: {advGenLoss:.4f}, Disc loss: {discLoss:.4f}, Label loss {labelLoss:.4f}'
        )

    def _check_discriminator_and_classifier(self, epoch, x_pl, x_u):
        if not self.use_original_paper_code:
            self.log.write("epoch: {}\n".format(epoch + 1))
        d_x_pu, d_x_u = self.model.check_disc(x_pl, x_u)
        d_x_pu2, d_x_pl2 = self.model.check_pn(x_pl, x_u)

        if not self.use_original_paper_code:
            self.log.write(
                "d_x_pu: {}, d_x_u: {}\n".format(torch.mean(d_x_pu), torch.mean(d_x_u))
            )
            self.log.write(
                "d_x_pu2: {}, d_x_pl2: {}\n".format(
                    torch.mean(d_x_pu2), torch.mean(d_x_pl2)
                )
            )

    def _train_target_classifier(self, epoch, x_pl, x_u, losses):
        if self.model_post_vae_training is None:
            print("Model saved post vae training!")
            self.model_post_vae_training = copy.deepcopy(self.model)

        if self.use_original_paper_code:
            l5 = self.model.train_step_pn(x_pl, x_u)
        else:
            # use x_u_full (all U samples) instead of x_u
            l5 = self.model.train_step_pn(x_pl, self.x_u_full)
        losses["Target classifier"].append(l5)
        return l5

    def _train_autoencoder(self, epoch, x_pl, x_u, losses):
        l3 = self.model.train_step_disc(x_pl, x_u)
        l1, l2, l4 = self.model.train_step_vae(x_pl, x_u, epoch)

        if np.isnan(l1):
            raise ValueError(f"Autoencoder loss was NaN in epoch {epoch + 1}")

        losses["ELBO"].append(l1)
        losses["Adversarial generation"].append(l2)
        losses["Discriminator"].append(l3)
        losses["Label"].append(l4)

        return l1, l2, l3, l4

    def _pretrain_autoencoder(self):
        self.preLoss1, self.preLoss2 = [], []

        pretrain_time = time.time()
        for epoch in range(self.config["num_epoch_pre"]):
            print(
                f'[PRE-TRAIN] Exp: {self.num_exp} / c = {self.config["base_label_frequency"]:.2f} / Epoch: {epoch + 1}'
            )

            start_time = time.time()
            lst_1 = []
            lst_2 = []

            for x_pl, x_u in zip(self.DL_pl, self.DL_u):
                x_pl, x_u = x_pl[0], x_u[0]
                l1 = self.model.pretrain(x_pl, x_u)
                if np.isnan(l1):
                    raise ValueError(
                        f"(PRETRAIN) Autoencoder loss was NaN in epoch {epoch + 1}"
                    )

                lst_1.append(l1)
                if self.config["bool_pn_pre"]:
                    l2 = self.model.train_step_pn_pre(x_pl, x_u)
                    lst_2.append(l2)

            self.preLoss1.append(sum(lst_1) / len(lst_1))
            if self.config["bool_pn_pre"]:
                self.preLoss2.append(sum(lst_2) / len(lst_2))

            end_time = time.time()
            print(
                "[PRE-TRAIN] Remaining time: {} sec".format(
                    (self.config["num_epoch_pre"] - epoch - 1) * (end_time - start_time)
                )
            )
            print(f"[PRE-TRAIN] VAE Loss: {sum(lst_1) / len(lst_1)}")
            if self.config["bool_pn_pre"]:
                print(f"[PRE-TRAIN] PN Loss: {sum(lst_2) / len(lst_2)}")

        self._plotLoss(
            self.preLoss1, os.path.join(self.config["directory"], "loss_pretrain.png")
        )
        if self.config["bool_pn_pre"]:
            self._plotLoss(
                self.preLoss2,
                os.path.join(self.config["directory"], "loss_pretrain_pn.png"),
            )
        print("PRE-TRAIN finish!")

    def _prepare_log_files(self):
        os.makedirs(self.config["directory"], exist_ok=True)
        if not self.use_original_paper_code:
            self.log = open(os.path.join(self.config["directory"], "log.txt"), "w")
            self.log.write("config\n")
            self.log.write(str(self.config))
            self.log.write("\n")
            self.log.close()

            self.log2 = open(os.path.join(self.config["directory"], "log_PN.txt"), "w")
            self.log2.write("config\n")
            self.log2.write(str(self.config))
            self.log2.write("\n")
            self.log2.close()

    def _prepare_dataloaders(self, vae_pu_data):
        (
            self.x_pl_full,
            self.y_pl_full,
            self.x_u_full,
            self.y_u_full,
            self.x_val,
            self.y_val,
            self.s_val,
            self.x_test,
            self.y_test,
            self.s_test,
        ) = vae_pu_data

        if "MNIST" in self.config["data"]:
            self.x_pl_full = (self.x_pl_full + 1.0) / 2.0
            self.x_u_full = (self.x_u_full + 1.0) / 2.0
            self.x_val = (self.x_val + 1.0) / 2.0
            self.x_test = (self.x_test + 1.0) / 2.0

        self.DL_pl = InfiniteDataLoader(
            TensorDataset(
                self.x_pl_full,
                torch.arange(len(self.x_pl_full)).to(self.config["device"]),
            ),
            batch_size=self.config["batch_size_l"],
            shuffle=True,
        )
        self.DL_u = DataLoader(
            TensorDataset(
                self.x_u_full,
                torch.arange(len(self.x_u_full)).to(self.config["device"]),
            ),
            batch_size=self.config["batch_size_u"],
            shuffle=True,
        )
        self.DL_u_full = DataLoader(
            TensorDataset(
                self.x_u_full,
                self.y_u_full,
                torch.arange(len(self.x_u_full)).to(self.config["device"]),
            ),
            batch_size=self.config["batch_size_u"],
            shuffle=True,
        )
        self.DL_val = DataLoader(
            TensorDataset(
                self.x_val,
                self.y_val,
                torch.arange(len(self.x_val)).to(self.config["device"]),
            ),
            batch_size=self.config["batch_size_val"],
            shuffle=True,
        )
        self.DL_val_s = DataLoader(
            TensorDataset(
                self.x_val,
                self.y_val,
                self.s_val,
                torch.arange(len(self.x_val)).to(self.config["device"]),
            ),
            batch_size=self.config["batch_size_val"],
            shuffle=True,
        )
        self.DL_test = DataLoader(
            TensorDataset(
                self.x_test,
                self.y_test,
                torch.arange(len(self.x_test)).to(self.config["device"]),
            ),
            batch_size=self.config["batch_size_test"],
            shuffle=True,
        )
        self.DL_test_s = DataLoader(
            TensorDataset(
                self.x_test,
                self.y_test,
                self.s_test,
                torch.arange(len(self.x_test)).to(self.config["device"]),
            ),
            batch_size=self.config["batch_size_test"],
            shuffle=True,
        )

    def _prepare_model(self):
        model_en = VAEencoder(self.config).to(self.config["device"])
        model_de = VAEdecoder(self.config).to(self.config["device"])
        model_disc = discriminator(self.config).to(self.config["device"])
        model_cl = classifier_o(self.config).to(self.config["device"])
        model_pn = classifier_pn(self.config).to(self.config["device"])

        opt_en = Adam(model_en.parameters(), lr=self.config["lr_pu"], eps=1e-07)
        opt_de = Adam(model_de.parameters(), lr=self.config["lr_pu"], eps=1e-07)
        opt_disc = Adam(
            model_disc.parameters(),
            lr=self.config["lr_disc"],
            betas=(self.config["beta1"], self.config["beta2"]),
            eps=1e-07,
        )
        opt_cl = Adam(
            model_cl.parameters(), lr=self.config["lr_pu"], weight_decay=1e-5, eps=1e-07
        )
        opt_pn = Adam(
            model_pn.parameters(), lr=self.config["lr_pn"], weight_decay=1e-5, eps=1e-07
        )

        self.model = myPU(
            self.config,
            model_en,
            model_de,
            model_disc,
            model_cl,
            model_pn,
            opt_en,
            opt_de,
            opt_disc,
            opt_cl,
            opt_pn,
        )

    def _load_trained_vae_pu(self):
        model = torch.load(os.path.join(self.config["directory"], "model_pre_occ.pt"))
        with open(
            os.path.join(self.config["directory"], "metric_values.json"), "r"
        ) as f:
            metric_values = json.load(f)
        (
            self.acc_pre_occ,
            self.precision_pre_occ,
            self.recall_pre_occ,
            self.f1_pre_occ,
        ) = (
            metric_values["Accuracy"],
            metric_values["Precision"],
            metric_values["Recall"],
            metric_values["F1 score"],
        )
        if "Time" in metric_values:
            self.baseline_training_time = metric_values["Time"]
        else:
            self.baseline_training_time = None
        print("Pre-OCC model loaded from file!")
        print(
            "final test pre-occ: acc: {0:.4f}, precision: {1:.4f}, recall: {2:.4f}, f1-score: {3:.4f}".format(
                self.acc_pre_occ,
                self.precision_pre_occ,
                self.recall_pre_occ,
                self.f1_pre_occ,
            )
        )
        return model

    def _plotLoss(self, lst, fname):
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, len(lst) + 1), lst)

        plt.savefig(fname)
        plt.close()
