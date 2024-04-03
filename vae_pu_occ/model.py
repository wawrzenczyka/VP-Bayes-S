import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _initialize_weights(module):
    # Initialize weights as in Keras
    for m in module.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight, gain=1)
            torch.nn.init.constant_(m.bias, 0)
        if isinstance(m, nn.BatchNorm1d):
            m.eval()


class VAEencoder(nn.Module):
    def __init__(self, config):
        super(VAEencoder, self).__init__()
        self.config = config

        self.vae_en_y = nn.Sequential()
        for i, (in_neurons, out_neurons) in enumerate(
            zip(
                [self.config["n_input"]] + self.config["n_hidden_vae_e"][:-1],
                self.config["n_hidden_vae_e"],
            )
        ):
            self.vae_en_y.add_module(f"linear_{i}", nn.Linear(in_neurons, out_neurons))
            self.vae_en_y.add_module(
                f"batch_norm_{i}", nn.BatchNorm1d(out_neurons, momentum=0.99, eps=0.001)
            )
            self.vae_en_y.add_module(f"leaky_relu_{i}", nn.LeakyReLU(0.2))

        self.vae_en_o = nn.Sequential()
        for i, (in_neurons, out_neurons) in enumerate(
            zip(
                [self.config["n_input"] + self.config["n_o"]]
                + self.config["n_hidden_vae_e"][:-1],
                self.config["n_hidden_vae_e"],
            )
        ):
            self.vae_en_o.add_module(f"linear_{i}", nn.Linear(in_neurons, out_neurons))
            self.vae_en_o.add_module(
                f"batch_norm_{i}", nn.BatchNorm1d(out_neurons, momentum=0.99, eps=0.001)
            )
            self.vae_en_o.add_module(f"leaky_relu_{i}", nn.LeakyReLU(0.2))

        self.vae_en_y_mu = nn.Sequential()
        self.vae_en_y_mu.add_module(
            f"linear_y_mu",
            nn.Linear(self.config["n_hidden_vae_e"][-1], self.config["n_h_y"]),
        )

        self.vae_en_y_lss = nn.Sequential()
        self.vae_en_y_lss.add_module(
            f"linear_y_lss",
            nn.Linear(self.config["n_hidden_vae_e"][-1], self.config["n_h_y"]),
        )

        self.vae_en_o_mu = nn.Sequential()
        self.vae_en_o_mu.add_module(
            f"linear_o_mu",
            nn.Linear(self.config["n_hidden_vae_e"][-1], self.config["n_h_o"]),
        )

        self.vae_en_o_lss = nn.Sequential()
        self.vae_en_o_lss.add_module(
            f"linear_o_lss",
            nn.Linear(self.config["n_hidden_vae_e"][-1], self.config["n_h_o"]),
        )

        _initialize_weights(self)

    def encode(self, x, o):
        # separate each component
        hidden_y = self.vae_en_y(x)
        y_mu = self.vae_en_y_mu(hidden_y)
        y_lss = self.vae_en_y_lss(hidden_y)

        hidden_o = self.vae_en_o(torch.cat([x, o], axis=1))
        o_mu = self.vae_en_o_mu(hidden_o)
        o_lss = self.vae_en_o_lss(hidden_o)

        return y_mu, y_lss, o_mu, o_lss

    def encode_y(self, x):
        hidden_y = self.vae_en_y(x)
        y_mu = self.vae_en_y_mu(hidden_y)
        y_lss = self.vae_en_y_lss(hidden_y)

        return y_mu, y_lss


class VAEdecoder(nn.Module):
    def __init__(self, config):
        super(VAEdecoder, self).__init__()
        self.config = config

        # separate each component
        self.vae_de = nn.Sequential()
        for i, (in_neurons, out_neurons) in enumerate(
            zip(
                [self.config["n_h_y"] + self.config["n_h_o"]]
                + self.config["n_hidden_vae_d"][:-1],
                self.config["n_hidden_vae_d"],
            )
        ):
            self.vae_de.add_module(f"linear_{i}", nn.Linear(in_neurons, out_neurons))
            self.vae_de.add_module(
                f"batch_norm_{i}", nn.BatchNorm1d(out_neurons, momentum=0.99, eps=0.001)
            )
            self.vae_de.add_module(f"leaky_relu_{i}", nn.LeakyReLU(0.2))
        self.vae_de.add_module(
            f"linear_out",
            nn.Linear(self.config["n_hidden_vae_d"][-1], self.config["n_input"]),
        )

        _initialize_weights(self)

    def decode(self, h_y, h_o, sigmoid=False):
        recon = self.vae_de(torch.cat([h_y, h_o], axis=1))
        if sigmoid:
            recon = torch.sigmoid(recon)
        return recon


class discriminator(nn.Module):
    def __init__(self, config):
        super(discriminator, self).__init__()
        self.config = config

        self.disc_u = nn.Sequential()
        for i, (in_neurons, out_neurons) in enumerate(
            zip(
                [self.config["n_input"]] + self.config["n_hidden_disc"][:-1],
                self.config["n_hidden_disc"],
            )
        ):
            self.disc_u.add_module(f"linear_{i}", nn.Linear(in_neurons, out_neurons))
            self.disc_u.add_module(f"leaky_relu_{i}", nn.LeakyReLU(0.2))
            self.disc_u.add_module(f"dropout_{i}", nn.Dropout(0.3))
        last_shape = (
            self.config["n_hidden_disc"][-1]
            if len(self.config["n_hidden_disc"]) > 0
            else self.config["n_input"]
        )
        self.disc_u.add_module(f"linear_out", nn.Linear(last_shape, 1))

        _initialize_weights(self)

    def discriminate(self, x, sigmoid=False):
        disc = self.disc_u(x)
        if sigmoid:
            disc = torch.sigmoid(disc)
        return disc


class classifier_o(nn.Module):
    def __init__(self, config):
        super(classifier_o, self).__init__()
        self.config = config

        self.classification = nn.Sequential()
        for i, (in_neurons, out_neurons) in enumerate(
            zip(
                [self.config["n_h_o"]] + self.config["n_hidden_cl"][:-1],
                self.config["n_hidden_cl"],
            )
        ):
            self.classification.add_module(
                f"linear_{i}", nn.Linear(in_neurons, out_neurons)
            )
            self.classification.add_module(
                f"batch_norm_{i}", nn.BatchNorm1d(out_neurons, momentum=0.99, eps=0.001)
            )
            self.classification.add_module(f"leaky_relu_{i}", nn.LeakyReLU(0.2))

        last_shape = (
            self.config["n_hidden_cl"][-1]
            if len(self.config["n_hidden_cl"]) > 0
            else self.config["n_h_o"]
        )
        self.classification.add_module(f"linear_out", nn.Linear(last_shape, 1))

        _initialize_weights(self)

    def classify(self, x, sigmoid=False):
        c = self.classification(x)
        if sigmoid:
            c = torch.sigmoid(c)
        return c


class classifier_pn(nn.Module):
    def __init__(self, config):
        super(classifier_pn, self).__init__()
        self.config = config

        self.classification = nn.Sequential()
        for i, (in_neurons, out_neurons) in enumerate(
            zip(
                [self.config["n_input"]] + self.config["n_hidden_pn"][:-1],
                self.config["n_hidden_pn"],
            )
        ):
            self.classification.add_module(
                f"linear_{i}", nn.Linear(in_neurons, out_neurons)
            )
            self.classification.add_module(
                f"batch_norm_{i}", nn.BatchNorm1d(out_neurons, momentum=0.99, eps=0.001)
            )
            self.classification.add_module(f"leaky_relu_{i}", nn.LeakyReLU(0.2))

        last_shape = (
            self.config["n_hidden_pn"][-1]
            if len(self.config["n_hidden_pn"]) > 0
            else self.config["n_input"]
        )
        self.classification.add_module(f"linear_out", nn.Linear(last_shape, 1))

        _initialize_weights(self)

    def classify(self, x, sigmoid=False):
        c = self.classification(x)
        if sigmoid:
            c = torch.sigmoid(c)
        return c


class myPU:
    def __init__(
        self,
        config,
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
        opt_param=None,
        use_original_paper_code=False,
    ):
        self.config = config

        self.model_en = model_en
        self.model_de = model_de
        self.model_disc = model_disc
        self.model_cl = model_cl
        self.model_pn = model_pn

        self.opt_en = opt_en
        self.opt_de = opt_de
        self.opt_disc = opt_disc
        self.opt_cl = opt_cl
        self.opt_pn = opt_pn

        self.use_original_paper_code = use_original_paper_code

    def reparameterization(self, mu, log_sig_sq):
        eps = torch.randn_like(mu)
        return mu + torch.exp(log_sig_sq / 2.0) * eps

    def concat_data(self, x_pl, x_u):
        x = torch.cat([x_pl, x_u], axis=0)

        o_pl = torch.cat(
            [
                torch.ones([x_pl.shape[0], 1]).to(self.config["device"]),
                torch.zeros([x_pl.shape[0], 1]).to(self.config["device"]),
            ],
            axis=1,
        )
        o_u = torch.cat(
            [
                torch.zeros([x_u.shape[0], 1]).to(self.config["device"]),
                torch.ones([x_u.shape[0], 1]).to(self.config["device"]),
            ],
            axis=1,
        )
        o = torch.cat([o_pl, o_u], axis=0)

        return x, o

    def pretrain(self, x_pl, x_u):
        x, o = self.concat_data(x_pl, x_u)
        n_features = x.shape[1]

        h_y_mu, h_y_log_sig_sq, h_o_mu, h_o_log_sig_sq = self.model_en.encode(x, o)
        h_y = self.reparameterization(h_y_mu, h_y_log_sig_sq)
        h_o = self.reparameterization(h_o_mu, h_o_log_sig_sq)
        if "MNIST" in self.config["data"]:
            recon_x = self.model_de.decode(h_y, h_o)
            loss = torch.mean(
                n_features * F.binary_cross_entropy_with_logits(input=recon_x, target=x)
            )
        elif "CIFAR" in self.config["data"]:
            recon_x_mu = self.model_de.decode(h_y, h_o)
            loss = torch.mean(torch.sum(0.5 * torch.square(x - recon_x_mu), axis=1))
        elif "STL" in self.config["data"]:
            recon_x_mu = self.model_de.decode(h_y, h_o)
            loss = torch.mean(torch.sum(0.5 * torch.square(x - recon_x_mu), axis=1))
        else:
            recon_x_mu = self.model_de.decode(h_y, h_o)
            loss = torch.mean(torch.sum(0.5 * torch.square(x - recon_x_mu), axis=1))

        self.opt_en.zero_grad()
        self.opt_de.zero_grad()

        loss.backward()

        self.opt_en.step()
        self.opt_de.step()

        return loss.detach().cpu().numpy()

    def findPrior(self, x_tr_l, x_tr_u):
        from sklearn.mixture import GaussianMixture

        o_pl = torch.cat(
            [
                torch.ones([x_tr_l.shape[0], 1]).to(self.config["device"]),
                torch.zeros([x_tr_l.shape[0], 1]).to(self.config["device"]),
            ],
            axis=1,
        )
        o_u = torch.cat(
            [
                torch.zeros([x_tr_u.shape[0], 1]).to(self.config["device"]),
                torch.ones([x_tr_u.shape[0], 1]).to(self.config["device"]),
            ],
            axis=1,
        )

        h_y_u_mu, h_y_u_log_sig_sq, _, _ = self.model_en.encode(x_tr_u, o_u)
        h_y_u = h_y_u_mu
        h_y_l_mu, h_y_l_log_sig_sq, _, _ = self.model_en.encode(x_tr_l, o_pl)
        h_y_l = h_y_l_mu

        h_y = torch.cat([h_y_u, h_y_l], axis=0)

        gmm = GaussianMixture(n_components=2, covariance_type="diag")
        gmm.fit(h_y.detach().cpu().numpy())

        self.p = torch.tensor(gmm.weights_[1]).to(self.config["device"])
        self.mu = torch.tensor(np.asarray(gmm.means_, dtype=np.float32)).to(
            self.config["device"]
        )
        self.var = torch.tensor(np.asarray(gmm.covariances_, dtype=np.float32)).to(
            self.config["device"]
        )

        c0 = torch.mean(
            -0.5 * torch.true_divide(torch.square(h_y_l - self.mu[0]), self.var[0])
            - 0.5 * torch.log(self.var[0] + 1e-9),
            axis=1,
        )
        c1 = torch.mean(
            -0.5 * torch.true_divide(torch.square(h_y_l - self.mu[1]), self.var[1])
            - 0.5 * torch.log(self.var[1] + 1e-9),
            axis=1,
        )

        num0 = torch.sum(torch.greater(c0 - c1, 0.0).int())
        frac0 = num0 / x_tr_l.shape[0]
        if frac0 > 0.5:
            self.p = torch.tensor(gmm.weights_[0]).to(self.config["device"])
            self.mu[0], self.mu[1] = torch.tensor(gmm.means_[1]).to(
                self.config["device"]
            ), torch.tensor(gmm.means_[0]).to(self.config["device"])
            self.var[0], self.var[1] = torch.tensor(gmm.covariances_[1]).to(
                self.config["device"]
            ), torch.tensor(gmm.covariances_[0]).to(self.config["device"])
        self.mu = torch.tensor(np.asarray(self.mu.cpu(), dtype=np.float32)).to(
            self.config["device"]
        )
        self.var = torch.tensor(np.asarray(self.var.cpu(), dtype=np.float32)).to(
            self.config["device"]
        )

    def train_step_vae(self, x_pl, x_u, epoch):
        alpha_gen = self.config["alpha_gen"]
        alpha_gen2 = self.config["alpha_gen2"]
        if self.config["pi_given"] == None:
            p = self.config["pi_pl"] + self.config["pi_pu"]
        else:
            p = self.config["pi_given"]
        p = torch.tensor(p).to(self.config["device"])

        x, o = self.concat_data(x_pl, x_u)
        n_features = x.shape[1]

        h_y_mu, h_y_log_sig_sq, h_o_mu, h_o_log_sig_sq = self.model_en.encode(x, o)
        h_y = self.reparameterization(h_y_mu, h_y_log_sig_sq)
        h_o = self.reparameterization(h_o_mu, h_o_log_sig_sq)

        c0 = (
            -0.5 * torch.true_divide(torch.square(h_y - self.mu[0]), self.var[0])
            - 0.5 * torch.log(torch.clamp_min(self.var[0], 1e-9))
            + torch.log(torch.clamp_min(1 - p, 1e-9))
        )
        c1 = (
            -0.5 * torch.true_divide(torch.square(h_y - self.mu[1]), self.var[1])
            - 0.5 * torch.log(torch.clamp_min(self.var[1], 1e-9))
            + torch.log(torch.clamp_min(p, 1e-9))
        )

        c0 = torch.sum(c0, axis=1, keepdim=True)
        c1 = torch.sum(c1, axis=1, keepdim=True)

        c = torch.cat([c0, c1], axis=1)
        c = F.softmax(c, dim=1)[:, 1].reshape(-1, 1)

        _, _, _, x_pu, _ = self.generate(x_pl, x_u, self.config["mode"])
        d_x_pu = self.model_disc.discriminate(x_pu, sigmoid=False)
        label = torch.ones_like(d_x_pu)

        gan_loss = alpha_gen * torch.mean(
            F.binary_cross_entropy_with_logits(input=d_x_pu, target=label)
        )

        if epoch <= self.config["num_epoch_step1"]:
            gan_loss2 = 0 * torch.mean(torch.zeros(1).to(self.config["device"]))
        else:
            d_x_pu2 = self.model_pn.classify(x_pu, sigmoid=False)
            gan_loss2 = alpha_gen2 * torch.mean(
                self.sigmoid_loss(d_x_pu2, torch.ones_like(d_x_pu2))
            )

        loss1_0 = -torch.sum(
            0.5
            * (
                torch.log(torch.clamp_min(self.var[0], 1e-9))
                + torch.true_divide(
                    torch.exp(h_y_log_sig_sq) + torch.square(h_y_mu - self.mu[0]),
                    self.var[0],
                )
            ),
            axis=1,
            keepdim=True,
        )
        loss1_1 = -torch.sum(
            0.5
            * (
                torch.log(torch.clamp_min(self.var[1], 1e-9))
                + torch.true_divide(
                    torch.exp(h_y_log_sig_sq) + torch.square(h_y_mu - self.mu[1]),
                    self.var[1],
                )
            ),
            axis=1,
            keepdim=True,
        )

        loss1 = torch.mean(torch.multiply(1 - c, loss1_0) + torch.multiply(c, loss1_1))
        loss2 = -torch.mean(
            torch.sum(0.5 * (torch.exp(h_o_log_sig_sq) + torch.square(h_o_mu)), axis=1)
        )

        if "MNIST" in self.config["data"]:
            recon_x = self.model_de.decode(h_y, h_o)
            loss3 = -torch.mean(
                n_features * F.binary_cross_entropy_with_logits(input=recon_x, target=x)
            )
        elif "CIFAR" in self.config["data"]:
            recon_x = self.model_de.decode(h_y, h_o)
            loss3 = self.config["alpha_test"] * -torch.mean(
                torch.sum(0.5 * torch.square(x - recon_x), axis=1)
            )
        elif "STL" in self.config["data"]:
            recon_x = self.model_de.decode(h_y, h_o)
            loss3 = self.config["alpha_test"] * -torch.mean(
                torch.sum(0.5 * torch.square(x - recon_x), axis=1)
            )
        else:
            recon_x = self.model_de.decode(h_y, h_o)
            loss3 = -torch.mean(torch.sum(0.5 * torch.square(x - recon_x), axis=1))

        loss4 = torch.mean(torch.sum(0.5 * (1 + h_y_log_sig_sq), axis=1))
        loss5 = torch.mean(torch.sum(0.5 * (1 + h_o_log_sig_sq), axis=1))

        loss6 = torch.mean(
            -c * torch.log((c / torch.clamp_min(p, 1e-9)) + 1e-9)
            - (1 - c) * torch.log(((1 - c) / torch.clamp_min(1 - p, 1e-9)) + 1e-9)
        )

        c_o = self.model_cl.classify(h_o, sigmoid=False)
        label = torch.unsqueeze(o[:, 0], 1)
        loss7 = -torch.mean(F.binary_cross_entropy_with_logits(input=c_o, target=label))

        vade_loss = -self.config["alpha_vade"] * (
            loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + loss7
        )

        loss_t1 = vade_loss + gan_loss + gan_loss2

        self.opt_en.zero_grad()
        self.opt_de.zero_grad()
        self.opt_cl.zero_grad()

        loss_t1.backward()

        self.opt_en.step()
        self.opt_de.step()
        self.opt_cl.step()

        return (
            vade_loss.detach().cpu().numpy(),
            gan_loss.detach().cpu().numpy(),
            gan_loss2.detach().cpu().numpy(),
        )

    def encode_y(self, x):
        x, _ = self.concat_data(
            x, torch.empty((0, x.shape[1])).to(self.config["device"])
        )

        h_y_mu, h_y_log_sig_sq = self.model_en.encode_y(x)
        h_y = self.reparameterization(h_y_mu, h_y_log_sig_sq)
        return h_y

    def get_pu_from_clustering(self, x_pl, x_u, storey_pi, clustering_metric):
        h_y_pl = self.encode_y(x_pl)
        h_y_u = self.encode_y(x_u)

        h_y_pl_2 = torch.sum(torch.square(h_y_pl), dim=1).reshape(1, -1)
        h_y_u_2 = torch.sum(torch.square(h_y_u), dim=1).reshape(-1, 1)  # !!!

        distance = torch.sqrt(h_y_u_2 - 2 * torch.matmul(h_y_u, h_y_pl.T) + h_y_pl_2)

        del h_y_pl, h_y_u, h_y_pl_2, h_y_u_2

        if clustering_metric == "mean":
            distance_metric = torch.mean(distance, dim=1)
        elif clustering_metric == "quantile":
            q = 0.1
            distance_metric = torch.quantile(distance, q, dim=1)
        elif clustering_metric == "quantmean":
            q = 0.1
            sorted_distances = torch.sort(distance, dim=1).values
            sorted_distances = sorted_distances[
                :, : round(q * sorted_distances.shape[1])
            ]
            distance_metric = torch.mean(sorted_distances, dim=1)
        else:
            raise NotImplementedError()

        x_u_sorted_idx = torch.argsort(distance_metric)
        n_pu_samples = int(round(storey_pi * len(x_u)))
        pu_indices = x_u_sorted_idx[:n_pu_samples]

        return x_u[pu_indices], pu_indices

    def get_pu_from_clustering_batched(
        self,
        x_pl,
        x_u_batch,
        n_pu_samples,
        clustering_metric,
        batch_start_idx=0,
        best_distances=None,
        best_xus=None,
        best_pu_indices=None,
    ):
        h_y_pl = self.encode_y(x_pl)
        h_y_u = self.encode_y(x_u_batch)

        h_y_pl_2 = torch.sum(torch.square(h_y_pl), dim=1).reshape(1, -1)
        h_y_u_2 = torch.sum(torch.square(h_y_u), dim=1).reshape(-1, 1)  # !!!

        distance = torch.sqrt(h_y_u_2 - 2 * torch.matmul(h_y_u, h_y_pl.T) + h_y_pl_2)

        del h_y_pl, h_y_u, h_y_pl_2, h_y_u_2

        if clustering_metric == "mean":
            distance_metric = torch.mean(distance, dim=1)
        elif clustering_metric == "quantile":
            q = 0.1
            distance_metric = torch.quantile(distance, q, dim=1)
        elif clustering_metric == "quantmean":
            q = 0.1
            sorted_distances = torch.sort(distance, dim=1).values
            sorted_distances = sorted_distances[
                :, : round(q * sorted_distances.shape[1])
            ]
            distance_metric = torch.mean(sorted_distances, dim=1)
        else:
            raise NotImplementedError()

        distance_metric = torch.cat([distance_metric, best_distances])

        x_u_sorted_idx = torch.argsort(distance_metric)
        pu_indices = x_u_sorted_idx[:n_pu_samples]

        pu_indices_batch = pu_indices[pu_indices < len(x_u_batch)]
        pu_indices_best = pu_indices[pu_indices >= len(x_u_batch)] - len(x_u_batch)

        best_distances = distance_metric[pu_indices]
        best_xus = torch.cat([x_u_batch[pu_indices_batch], best_xus[pu_indices_best]])
        best_pu_indices = torch.cat(
            [pu_indices_batch + batch_start_idx, best_pu_indices[pu_indices_best]]
        )

        del (
            distance,
            distance_metric,
            x_u_sorted_idx,
            pu_indices,
            pu_indices_batch,
            pu_indices_best,
        )

        return best_xus, best_pu_indices, best_distances

    def vae_predict_proba(self, x):
        if self.config["pi_given"] == None:
            p = self.config["pi_pl"] + self.config["pi_pu"]
        else:
            p = self.config["pi_given"]
        p = torch.tensor(p).to(self.config["device"])

        h_y_mu, h_y_log_sig_sq = self.model_en.encode_y(x)
        h_y = self.reparameterization(h_y_mu, h_y_log_sig_sq)

        c0 = (
            -0.5 * torch.true_divide(torch.square(h_y - self.mu[0]), self.var[0])
            - 0.5 * torch.log(torch.clamp_min(self.var[0], 1e-9))
            + torch.log(torch.clamp_min(1 - p, 1e-9))
        )
        c1 = (
            -0.5 * torch.true_divide(torch.square(h_y - self.mu[1]), self.var[1])
            - 0.5 * torch.log(torch.clamp_min(self.var[1], 1e-9))
            + torch.log(torch.clamp_min(p, 1e-9))
        )

        c0 = torch.sum(c0, axis=1, keepdim=True)
        c1 = torch.sum(c1, axis=1, keepdim=True)

        c = torch.cat([c0, c1], axis=1)
        c = F.softmax(c, dim=1)
        return c[:, 1]

    def vae_predict(self, x):
        proba = self.vae_predict_proba(x)
        return torch.where(proba > 0.5, 1, -1)

    def generate(self, x_pl, x_u, mode="near_o"):
        if mode == "near_o":
            # nearest h_o
            o_pl = (
                torch.tensor(
                    np.concatenate(
                        [np.ones([x_pl.shape[0], 1]), np.zeros([x_pl.shape[0], 1])],
                        axis=1,
                    )
                )
                .float()
                .to(self.config["device"])
            )
            o_u = (
                torch.tensor(
                    np.concatenate(
                        [np.zeros([x_u.shape[0], 1]), np.ones([x_u.shape[0], 1])],
                        axis=1,
                    )
                )
                .float()
                .to(self.config["device"])
            )

            h_y_mu, h_y_log_sig_sq, h_o_mu, h_o_log_sig_sq = self.model_en.encode(
                x_pl, o_pl
            )
            h_y = self.reparameterization(h_y_mu, h_y_log_sig_sq)
            h_o = self.reparameterization(h_o_mu, h_o_log_sig_sq)

            _, _, h_o_mu_x, h_o_log_sig_sq_x = self.model_en.encode(x_u, o_u)
            h_o_x = self.reparameterization(h_o_mu_x, h_o_log_sig_sq_x)

            h_o_2 = torch.sum(torch.square(h_o), dim=1)
            h_o_x_2 = torch.sum(torch.square(h_o_x), dim=1)

            h_o_2 = h_o_2.reshape(-1, 1)
            h_o_x_2 = h_o_x_2.reshape(1, -1)

            distance = torch.sqrt(h_o_2 - 2 * torch.matmul(h_o, h_o_x.T) + h_o_x_2)
            lstIdx = torch.argmin(distance, dim=1)
            ne_h_o = h_o_x[lstIdx]

            x_u_select = x_u[lstIdx]

            if "MNIST" in self.config["data"]:
                x_pu = self.model_de.decode(h_y, ne_h_o, sigmoid=True)
            else:
                x_pu_mu = self.model_de.decode(h_y, ne_h_o)
                # x_pu = self.reparameterization(x_pu_mu, x_pu_lss)
                x_pu = x_pu_mu
            return h_y, h_o, ne_h_o, x_pu, x_u_select

        elif mode == "near_y":
            o_pl = (
                torch.tensor(
                    np.concatenate(
                        [np.ones([x_pl.shape[0], 1]), np.zeros([x_pl.shape[0], 1])],
                        axis=1,
                    )
                )
                .float()
                .to(self.config["device"])
            )
            o_u = (
                torch.tensor(
                    np.concatenate(
                        [np.zeros([x_u.shape[0], 1]), np.ones([x_u.shape[0], 1])],
                        axis=1,
                    )
                )
                .float()
                .to(self.config["device"])
            )

            h_y_mu, h_y_log_sig_sq, h_o_mu, h_o_log_sig_sq = self.model_en.encode(
                x_pl, o_pl
            )
            h_y = self.reparameterization(h_y_mu, h_y_log_sig_sq)
            h_o = self.reparameterization(h_o_mu, h_o_log_sig_sq)

            (
                h_y_mu_x,
                h_y_log_sig_sq_x,
                h_o_mu_x,
                h_o_log_sig_sq_x,
            ) = self.model_en.encode(x_u, o_u)
            h_y_x = self.reparameterization(h_y_mu_x, h_y_log_sig_sq_x)
            h_o_x = self.reparameterization(h_o_mu_x, h_o_log_sig_sq_x)

            h_y_2 = torch.sum(torch.square(h_y), 1)
            h_y_x_2 = torch.sum(torch.square(h_y_x), 1)

            h_y_2 = h_y_2.reshape(-1, 1)
            h_y_x_2 = h_y_x_2.reshape(1, -1)

            distance = torch.sqrt(h_y_2 - 2 * torch.matmul(h_y, h_y_x.T) + h_y_x_2)
            lstIdx = torch.argmin(distance, dim=1)
            ne_h_o = h_o_x[lstIdx]

            x_u_select = x_u[lstIdx]

            if "MNIST" in self.config["data"]:
                x_pu = self.model_de.decode(h_y, ne_h_o, sigmoid=True)
            else:
                x_pu_mu = self.model_de.decode(h_y, ne_h_o)
                x_pu = x_pu_mu
            return h_y, h_o, ne_h_o, x_pu, x_u_select

        elif mode == "random":
            x_pu = x_pl
            return 0, 0, 0, x_pu, 0

        else:
            NotImplementedError()

    def train_step_disc(self, x_pl, x_u):
        alpha_disc = self.config["alpha_disc"]
        alpha_disc2 = self.config["alpha_disc2"]

        _, _, _, x_pu, _ = self.generate(x_pl, x_u, self.config["mode"])

        d_x_pu = self.model_disc.discriminate(x_pu, sigmoid=False)
        d_x_u = self.model_disc.discriminate(x_u, sigmoid=False)

        label_pu = torch.zeros_like(d_x_pu).to(self.config["device"])
        label_u = torch.ones_like(d_x_u).to(self.config["device"])

        disc_loss1 = alpha_disc * (
            torch.mean(
                F.binary_cross_entropy_with_logits(input=d_x_pu, target=label_pu)
            )
            + torch.mean(
                F.binary_cross_entropy_with_logits(input=d_x_u, target=label_u)
            )
        )

        disc_loss = disc_loss1

        self.opt_disc.zero_grad()
        disc_loss.backward()
        self.opt_disc.step()

        return disc_loss.detach().cpu().numpy()

    def sigmoid_loss(self, t, y):
        return torch.sigmoid(-t * y)

    def logistic_loss(self, t, y):
        return torch.softplus(-t * y)

    def train_step_pn_pre(self, x_pl, x_u):
        pi_pl = self.config["pi_pl"]
        pi_pu = self.config["pi_pu"]
        pi_u = self.config["pi_u"]

        pi_p = pi_pl + pi_pu

        pn_x_pl = self.model_pn.classify(x_pl, sigmoid=False)
        pn_x_u = self.model_pn.classify(x_u, sigmoid=False)

        pu1_loss = torch.mean(self.sigmoid_loss(pn_x_pl, torch.ones_like(pn_x_pl)))
        pu2_loss = torch.mean(
            -pi_p * self.sigmoid_loss(pn_x_pl, -torch.ones_like(pn_x_pl))
        )
        u_loss = torch.mean(pi_u * self.sigmoid_loss(pn_x_u, -torch.ones_like(pn_x_u)))
        if torch.greater_equal(pu2_loss + u_loss, 0):
            pn_loss = pu1_loss + pu2_loss + u_loss
        else:
            pn_loss = -(pu2_loss + u_loss)

        self.opt_pn.zero_grad()
        pn_loss.backward()

        return pn_loss.detach().cpu().numpy()

    def train_step_pn(self, x_pl, x_u):
        pi_pl = self.config["pi_pl"]
        pi_pu = self.config["pi_pu"]
        pi_u = self.config["pi_u"]

        with torch.no_grad():
            _, _, _, x_pu, _ = self.generate(x_pl, x_u, self.config["mode"])

        pn_x_pl = self.model_pn.classify(x_pl, sigmoid=False)
        pn_x_pu = self.model_pn.classify(x_pu, sigmoid=False)
        pn_x_u = self.model_pn.classify(x_u, sigmoid=False)

        if self.use_original_paper_code:
            pl_loss = pi_pl * torch.mean(
                self.sigmoid_loss(pn_x_pl, torch.ones_like(pn_x_pl))
            )
            pu1_loss = pi_pu * torch.mean(
                self.sigmoid_loss(pn_x_pu, torch.ones_like(pn_x_pu))
            )
            pu2_loss = -pi_pu * torch.mean(
                self.sigmoid_loss(pn_x_pu, -torch.ones_like(pn_x_pu))
            )
            u_loss = pi_u * torch.mean(
                self.sigmoid_loss(pn_x_u, -torch.ones_like(pn_x_u))
            )
        else:
            # logistic loss
            pl_loss = -torch.mean(pi_pl * F.logsigmoid(pn_x_pl))
            pu1_loss = -torch.mean(pi_pu * F.logsigmoid(pn_x_pu))
            pu2_loss = -torch.mean(-pi_pu * F.logsigmoid(-pn_x_pu))
            u_loss = -torch.mean(pi_u * F.logsigmoid(-pn_x_u))

        positive_risk = pl_loss + pu1_loss
        negative_risk = pu2_loss + u_loss
        pn_loss = (
            positive_risk + negative_risk
        )  # gradient will be calculated based on this value

        loss_value = pn_loss  # this value will be returned for display

        beta = self.config["nnPU_beta"]
        gamma = self.config["nnPU_gamma"]

        if negative_risk < 0:
            if beta is not None and gamma is not None:
                if negative_risk < -beta:
                    loss_value = positive_risk - beta
                    pn_loss = -gamma * negative_risk
            else:
                # we assume beta == 0, gamma == 1
                loss_value = positive_risk
                pn_loss = -negative_risk

        self.opt_pn.zero_grad()
        pn_loss.backward()
        self.opt_pn.step()

        return loss_value.detach().cpu().numpy()

    def train_step_pn_true_x_pu(self, x_pl, x_u, x_pu, pi_pl, pi_u, pi_pu):
        pn_x_pl = self.model_pn.classify(x_pl, sigmoid=False)
        pn_x_pu = self.model_pn.classify(x_pu, sigmoid=False)
        pn_x_u = self.model_pn.classify(x_u, sigmoid=False)

        pl_loss = -torch.mean(pi_pl * F.logsigmoid(pn_x_pl))
        pu1_loss = -torch.mean(pi_pu * F.logsigmoid(pn_x_pu))
        pu2_loss = -torch.mean(-pi_pu * F.logsigmoid(-pn_x_pu))
        u_loss = -torch.mean(pi_u * F.logsigmoid(-pn_x_u))

        if torch.greater_equal(pu2_loss + u_loss, 0):
            pn_loss = pl_loss + pu1_loss + pu2_loss + u_loss
        else:
            print("--- Unexpected! ---")
            pn_loss = pl_loss + pu1_loss

        self.opt_pn.zero_grad()
        pn_loss.backward()
        self.opt_pn.step()

        return pn_loss.detach().cpu().numpy()

    def compare(self, fname, x_pl, x_u, n=1):
        lst_x_pu = []
        plt.figure(figsize=(10, 10))

        for _ in range(3):
            h_y, h_o, ne_h_o, x_pu, _ = self.generate(x_pl, x_u, self.config["mode"])
            lst_x_pu.append(x_pu)

        recon_x_pl = self.model_de.decode(h_y, h_o, sigmoid=True)

        if "MNIST" in self.config["data"]:
            for i in range(1, n + 1):
                plt.subplot(n, 5, (i - 1) * 5 + 1)
                plt.imshow(
                    x_pl[i - 1].detach().cpu().numpy().reshape(28, 28),
                    vmin=0,
                    vmax=1,
                    cmap="gray",
                )
                plt.title("real")

                plt.subplot(n, 5, (i - 1) * 5 + 2)
                plt.imshow(
                    recon_x_pl[i - 1].detach().cpu().numpy().reshape(28, 28),
                    vmin=0,
                    vmax=1,
                    cmap="gray",
                )
                plt.title("fake_pl")

                plt.subplot(n, 5, (i - 1) * 5 + 3)
                plt.imshow(
                    lst_x_pu[0][i - 1].detach().cpu().numpy().reshape(28, 28),
                    vmin=0,
                    vmax=1,
                    cmap="gray",
                )
                plt.title("fake_pu1")

                plt.subplot(n, 5, (i - 1) * 5 + 4)
                plt.imshow(
                    lst_x_pu[1][i - 1].detach().cpu().numpy().reshape(28, 28),
                    vmin=0,
                    vmax=1,
                    cmap="gray",
                )
                plt.title("fake_pu2")

                plt.subplot(n, 5, (i - 1) * 5 + 5)
                plt.imshow(
                    lst_x_pu[2][i - 1].detach().cpu().numpy().reshape(28, 28),
                    vmin=0,
                    vmax=1,
                    cmap="gray",
                )
                plt.title("fake_pu3")
        else:
            for i in range(1, n + 1):
                plt.subplot(n, 5, (i - 1) * 5 + 1)
                plt.imshow(
                    x_pl[i - 1].detach().cpu().numpy(), vmin=0, vmax=1, cmap="gray"
                )
                plt.title("real")

                plt.subplot(n, 5, (i - 1) * 5 + 2)
                plt.imshow(
                    recon_x_pl[i - 1].detach().cpu().numpy(),
                    vmin=0,
                    vmax=1,
                    cmap="gray",
                )
                plt.title("fake_pl")

                plt.subplot(n, 5, (i - 1) * 5 + 3)
                plt.imshow(
                    lst_x_pu[0][i - 1].detach().cpu().numpy(),
                    vmin=0,
                    vmax=1,
                    cmap="gray",
                )
                plt.title("fake_pu1")

                plt.subplot(n, 5, (i - 1) * 5 + 4)
                plt.imshow(
                    lst_x_pu[1][i - 1].detach().cpu().numpy(),
                    vmin=0,
                    vmax=1,
                    cmap="gray",
                )
                plt.title("fake_pu2")

                plt.subplot(n, 5, (i - 1) * 5 + 5)
                plt.imshow(
                    lst_x_pu[2][i - 1].detach().cpu().numpy(),
                    vmin=0,
                    vmax=1,
                    cmap="gray",
                )
                plt.title("fake_pu3")

        plt.savefig(fname)
        plt.close()

    def check_disc(self, x_pl, x_u):
        _, _, _, x_pu, _ = self.generate(x_pl, x_u, self.config["mode"])

        d_x_pu = self.model_disc.discriminate(x_pu, sigmoid=True)
        d_x_u = self.model_disc.discriminate(x_u, sigmoid=True)

        return d_x_pu, d_x_u

    def check_pn(self, x_pl, x_u):
        _, _, _, x_pu, _ = self.generate(x_pl, x_u, self.config["mode"])

        d_x_pu = self.model_pn.classify(x_pu, sigmoid=True)
        d_x_pl = self.model_pn.classify(x_pl, sigmoid=True)

        return d_x_pu, d_x_pl

    def accuracy(self, dataset, use_vae=False):
        tp = 0
        tn = 0
        fp = 0
        fn = 0

        for x_val, y_val, idx in dataset:
            if use_vae:
                c = self.vae_predict(x_val)
            else:
                c = self.model_pn.classify(x_val, sigmoid=False)

            tp = (
                tp
                + torch.sum(torch.greater(c[y_val == 1], 0).int())
                .detach()
                .cpu()
                .numpy()
            )
            fn = (
                fn
                + torch.sum(torch.less_equal(c[y_val == 1], 0).int())
                .detach()
                .cpu()
                .numpy()
            )
            fp = (
                fp
                + torch.sum(torch.greater(c[y_val == -1], 0).int())
                .detach()
                .cpu()
                .numpy()
            )
            tn = (
                tn
                + torch.sum(torch.less_equal(c[y_val == -1], 0).int())
                .detach()
                .cpu()
                .numpy()
            )

        if tp + fp == 0:
            precision = 0
        else:
            precision = tp / (tp + fp)
        if tp + fn == 0:
            recall = 0
        else:
            recall = tp / (tp + fn)

        acc = (tp + tn) / (tp + tn + fp + fn)
        f1 = 2 * precision * recall / (precision + recall)

        return acc, precision, recall, f1

    def loss_val(self, x_pl, x_u):
        pi_pl = self.config["pi_pl"]
        pi_pu = self.config["pi_pu"]
        pi_u = self.config["pi_u"]

        pi_p = pi_pl + pi_pu

        _, _, _, x_pu, _ = self.generate(x_pl, x_u, self.config["mode"])

        pn_x_pl = self.model_pn.classify(x_pl, sigmoid=False)
        pn_x_pu = self.model_pn.classify(x_pu, sigmoid=False)
        pn_x_u = self.model_pn.classify(x_u, sigmoid=False)

        pl_loss = -torch.mean(pi_pl * F.logsigmoid(pn_x_pl))
        pu1_loss = -torch.mean(pi_pu * F.logsigmoid(pn_x_pu))
        pu2_loss = -torch.mean(-pi_pu * F.logsigmoid(-pn_x_pu))
        u_loss = -torch.mean(pi_u * F.logsigmoid(-pn_x_u))

        return (pl_loss + pu1_loss + pu2_loss + u_loss).detach().cpu().numpy()

    def check_cl(self, x_pl, x_u):
        np.random.shuffle(x_pl)
        np.random.shuffle(x_u)
        x_pl = x_pl[:10]
        x_u = x_u[:10]

        _, h_o, ne_h_o, _ = self.generate(x_pl, x_u)

        c_h_o = self.model_cl.classify(h_o, sigmoid=True)
        c_ne_h_o = self.model_cl.classify(ne_h_o, sigmoid=True)

        _, _, h_o_mu_2, h_o_log_sig_sq_2 = self.model_en.encode(x_u)
        h_o_2 = self.reparameterization(h_o_mu_2, h_o_log_sig_sq_2)

        c_h_o_2 = self.model_cl.classify(h_o_2, sigmoid=True)

        return c_h_o, c_ne_h_o, c_h_o_2
