import argparse
import copy

from .dataset.dataset_custom import get_custom_loaders as get_loaders
from .dataset.dataset_custom import get_custom_test_loader
from .model.model_fashionmnist import NetworkPhi
from .vpu import *


class VPU:
    def __init__(
        self,
        representation="DV",  # "DV" / "NJW" / "no-name"
        pi=None,
        normalize_phi=False,
        use_extra_penalty=False,
        extra_penalty_config=None,
        batch_size=512,
    ) -> None:
        self.config = argparse.Namespace(
            dataset="fashionMNIST",
            gpu=0,
            val_iterations=30,
            batch_size=batch_size,
            # learning_rate=3e-5,
            learning_rate=3e-4,
            epochs=50,
            mix_alpha=0.3,
            # lam=0.03,
            lam=0.3,
            num_labeled=3000,
            positive_label_list=[1, 4, 7],
        )

        self.pi = pi
        self.representation = representation
        self.normalize_phi = normalize_phi
        self.use_extra_penalty = use_extra_penalty
        self.extra_penalty_config = extra_penalty_config

    def train(self, x_l, x_u):
        x_loader, p_loader, val_x_loader, val_p_loader = get_loaders(
            x_l, x_u, batch_size=self.config.batch_size
        )

        lowest_val_var = math.inf  # lowest variational loss on validation set

        if self.representation == "new-variant":
            self.model_phi = NetworkPhi(use_softmax=False)
        else:
            self.model_phi = NetworkPhi()

        if torch.cuda.is_available():
            self.model_phi = self.model_phi.cuda()

        # set up the optimizer
        lr_phi = self.config.learning_rate
        opt_phi = torch.optim.Adam(
            self.model_phi.parameters(), lr=lr_phi, betas=(0.5, 0.99)
        )

        for epoch in range(self.config.epochs):
            # adjust the optimizer
            if epoch % 20 == 19:
                lr_phi /= 2
                opt_phi = torch.optim.Adam(
                    self.model_phi.parameters(), lr=lr_phi, betas=(0.5, 0.99)
                )

            # train the model \Phi
            phi_loss, var_loss, reg_loss, phi_p_mean, phi_x_mean = train(
                self.config,
                self.model_phi,
                opt_phi,
                p_loader,
                x_loader,
                self.representation,
                self.pi,
                self.use_extra_penalty,
                self.extra_penalty_config,
            )

            # max_phi is needed for normalization
            self.log_max_phi = -math.inf
            for idx, (data, _) in enumerate(x_loader):
                if torch.cuda.is_available():
                    data = data.cuda()

                candidate_log_max_phi = self.model_phi(data)[:, 1].max()

                if self.representation == "DV":
                    pass
                elif self.representation == "NJW":
                    candidate_log_max_phi += np.log(self.pi) - 1
                elif self.representation == "no-name":
                    candidate_max_phi = torch.exp(candidate_log_max_phi)

                    # max(phi, 0) * pi
                    candidate_max_phi = torch.maximum(
                        candidate_max_phi,
                        torch.zeros_like(candidate_max_phi).to(
                            candidate_log_max_phi.device
                        ),
                    ) * torch.from_numpy(self.pi)
                    # candidate_max_phi = (
                    #     1 - 1 / (torch.tensor(self.pi) + 1)
                    # ) * candidate_max_phi

                    candidate_log_max_phi = torch.log(candidate_max_phi)
                elif self.representation == "new-variant":
                    candidate_log_max_phi += np.log(self.pi)  # * pi

                self.log_max_phi = max(self.log_max_phi, candidate_log_max_phi)

            # evaluate the model \Phi
            val_var = evaluate_val(
                self.model_phi,
                x_loader,
                val_p_loader,
                val_x_loader,
                epoch,
                phi_loss,
                var_loss,
                reg_loss,
            )

            # assessing performance of the current model and decide whether to save it
            is_val_var_lowest = val_var < lowest_val_var
            lowest_val_var = min(lowest_val_var, val_var)

            if is_val_var_lowest:
                epoch_of_best_val = epoch

                self.best_model = copy.deepcopy(self.model_phi)
                self.best_log_max_phi = self.log_max_phi.clone().detach()
            else:
                if epoch > epoch_of_best_val + 10:
                    break

        print(f"Early stopping at epoch {epoch_of_best_val}")
        self.model_phi = self.best_model
        self.log_max_phi = self.best_log_max_phi

    def predict(self, x):
        test_loader = get_custom_test_loader(x)

        # feed test set to the model and calculate accuracy and AUC
        with torch.no_grad():
            for idx, (data,) in enumerate(test_loader):
                if torch.cuda.is_available():
                    data = data.cuda()
                log_phi = self.model_phi(data)[:, 1]

                if self.normalize_phi:
                    if self.representation == "DV":
                        log_phi -= self.log_max_phi
                    elif self.representation == "NJW":
                        log_phi += np.log(self.pi) - 1  # * pi / e
                        log_phi -= self.log_max_phi
                    elif self.representation == "no-name":
                        phi = torch.exp(log_phi)

                        # max(phi, 0) * pi
                        phi = torch.maximum(
                            phi, torch.zeros_like(phi)
                        ) * torch.from_numpy(self.pi)
                        # phi = (1 - 1 / (torch.tensor(self.pi) + 1)) * phi

                        log_phi = torch.log(phi)
                        log_phi -= self.log_max_phi
                    elif self.representation == "new-variant":
                        log_phi += np.log(self.pi)  # * pi

                        phi = log_phi.exp()
                        phi = phi.exp() * torch.tensor(self.pi)  # s = e^s * pi

                        # phi /= self.log_max_phi.exp().exp()

                        log_phi = phi.log()
                    elif "DistPU" in self.representation:
                        pass
                    else:
                        raise NotImplementedError("Unknown MixUp representation")
                if idx == 0:
                    log_phi_all = log_phi
                else:
                    log_phi_all = torch.cat((log_phi_all, log_phi))
        pred_all = np.array((log_phi_all > math.log(0.5)).cpu().detach())
        return pred_all
