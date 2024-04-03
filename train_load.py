import numpy as np
from config import config
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torch
from data_loading.vae_pu_dataloaders import get_dataset, create_vae_pu_adapter


def analysis(model, idx):
    model_config = model.config

    np.random.seed(idx)
    torch.manual_seed(idx)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_samples, val_samples, test_samples, label_frequency, pi_p, n_input = \
        get_dataset(config['data'], device, config['label_frequency'])
    vae_pu_data = \
        create_vae_pu_adapter(train_samples, val_samples, test_samples, device)
    x_tr_l, y_tr_l, x_tr_u, y_tr_u, x_val, y_val, x_te, y_te = vae_pu_data

    # clustering of h_y
    o1 = torch.cat([torch.ones([x_tr_l.shape[0], 1]), torch.zeros([x_tr_l.shape[0], 1])], dim=1).to(model_config['device'])
    o2 = torch.cat([torch.zeros([x_tr_u.shape[0], 1]), torch.ones([x_tr_u.shape[0], 1])], dim=1).to(model_config['device'])
    o = torch.cat([o1, o2], axis=0)
    test_tsne(model, model_config, torch.cat([x_tr_l, x_tr_u], dim=0), torch.cat([0.5*torch.ones_like(y_tr_l[:]), y_tr_u[:]], dim=0), o, 'train_tsne_add_obs_h_o', '$h_o$', mode='h_o')
    test_tsne(model, model_config, torch.cat([x_tr_l, x_tr_u], dim=0), torch.cat([0.5*torch.ones_like(y_tr_l[:]), y_tr_u[:]], dim=0), o, 'train_tsne_add_obs_h_y', '$h_y$', mode='h_y')

def test_tsne(model, model_config, x, y, o, fname, title, makecolor=False, mode='h_y'):

    if makecolor == False:
        color_num = y.detach().cpu().numpy()
        label_map = {
            -1: ('N', 0.1, '#1f77b4', 'o'),
            1: ('PU', 0.1, '#ff7f0e', 'o'),
            0.5: ('PL', 1, '#9467bd', 's')
        }
    else:
        color_num = []
        for i in range(len(y)):
            if y[i] == 3:
                color_num.append('g')
            if y[i] == 2:
                color_num.append('r')
            if y[i] == 1:
                color_num.append('y')
            if y[i] == -1:
                color_num.append('m')
            if y[i] == -2:
                color_num.append('b')
        label_map = { c: (c, 0.1) for c in color_num }


    h_y_mu, h_y_log_sig_sq, h_o_mu, h_o_log_sig_sq = model.model_en.encode(x, o)

    h_y_mu, h_y_log_sig_sq, h_o_mu, h_o_log_sig_sq = \
        h_y_mu.detach().cpu().numpy(), h_y_log_sig_sq.detach().cpu().numpy(), \
        h_o_mu.detach().cpu().numpy(), h_o_log_sig_sq.detach().cpu().numpy()

    tsne = TSNE(n_components=2)

    if mode == 'h_y':
        trans = tsne.fit_transform(h_y_mu)
    elif mode == 'h_o':
        if model_config['n_h_o'] == 2:
            trans = h_o_mu
        else:
            trans = tsne.fit_transform(h_o_mu)
    else:
        NotImplementedError()

    plt.figure()
    for l, val in label_map.items():
        (label, size, color, marker) = val
        trans_filtered = trans[color_num == l]
        scatter = plt.scatter(trans_filtered[:, 0], trans_filtered[:, 1], s=size, label=label, c=color, marker=marker)
    plt.legend()
    plt.title(title)

    plt.savefig(model_config['directory'] + fname + '.png', bbox_inches='tight', dpi=600)
    plt.close()
