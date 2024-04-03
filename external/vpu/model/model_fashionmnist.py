# model for FashionMNIST

import torch
import torch.nn as nn


class NetworkPhi(nn.Module):
    def __init__(self, use_softmax=True):
        self.use_softmax = use_softmax

        super(NetworkPhi, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.conv3 = nn.Conv2d(16, 120, kernel_size=5)
        self.bn_conv1 = nn.BatchNorm2d(6)
        self.bn_conv2 = nn.BatchNorm2d(16)
        self.mp = nn.MaxPool2d(2)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(120, 84)
        self.bn_fc1 = nn.BatchNorm1d(84)

        # self.fc2 = nn.Linear(84, 2)

        if self.use_softmax:
            self.fc2 = nn.Linear(84, 2)
            self.LogSoftMax = nn.LogSoftmax(dim=1)
        else:
            self.fc2 = nn.Linear(84, 1)

    def forward(self, x):
        in_size = x.size(0)
        out = self.relu(self.mp(self.conv1(x)))
        out = self.relu(self.mp(self.conv2(out)))
        out = self.relu(self.conv3(out))
        out = out.view(in_size, -1)
        out = self.relu(self.bn_fc1(self.fc1(out)))
        out = self.fc2(out)

        if self.use_softmax:
            out = self.LogSoftMax(out)
        else:
            # add dummy output
            out = torch.concat(
                [
                    torch.zeros_like(out, device=out.device),
                    out,
                ],
                dim=1,
            )
        return out
