# %%
import argparse

from dataset.dataset_fashionmnist import get_fashionMNIST_loaders as get_loaders
from model.model_fashionmnist import NetworkPhi
from vpu import *

args = argparse.Namespace(
    dataset="fashionMNIST",
    gpu=0,
    val_iterations=30,
    batch_size=500,
    learning_rate=3e-5,
    epochs=50,
    mix_alpha=0.3,
    lam=0.03,
    num_labeled=3000,
    positive_label_list=[1, 4, 7],
)


def main(config):
    # set up cuda if it is available
    if torch.cuda.is_available():
        torch.cuda.set_device(config.gpu)

    # set up the loaders
    if config.dataset in ["cifar10", "fashionMNIST", "stl10"]:
        x_loader, p_loader, val_x_loader, val_p_loader, test_loader, idx = get_loaders(
            batch_size=config.batch_size,
            num_labeled=config.num_labeled,
            positive_label_list=config.positive_label_list,
        )
    elif config.dataset in ["avila", "pageblocks", "grid"]:
        x_loader, p_loader, val_x_loader, val_p_loader, test_loader = get_loaders(
            batch_size=config.batch_size,
            num_labeled=config.num_labeled,
            positive_label_list=config.positive_label_list,
        )
    loaders = (p_loader, x_loader, val_p_loader, val_x_loader, test_loader)

    # please read the following information to make sure it is running with the desired setting
    print("==> Preparing data")
    print("    # train data: ", len(x_loader.dataset))
    print("    # labeled train data: ", len(p_loader.dataset))
    print("    # test data: ", len(test_loader.dataset))
    print("    # val x data:", len(val_x_loader.dataset))
    print("    # val p data:", len(val_p_loader.dataset))

    # something about saving the model
    checkpoint = get_checkpoint_path(config)
    if not os.path.isdir(checkpoint):
        mkdir_p(checkpoint)

    # call VPU
    run_vpu(config, loaders, NetworkPhi)


if __name__ == "__main__":
    main(args)
