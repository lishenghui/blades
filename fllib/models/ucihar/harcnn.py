import torch
import torch.nn as nn


# https://github.com/jindongwang/Deep-learning-activity-recognition/blob/master/pytorch/network.py # noqa: E50
class HARCNN(nn.Module):
    def __init__(
        self,
        in_channels=9,
        dim_hidden=64 * 26,
        num_classes=6,
        conv_kernel_size=(1, 9),
        pool_kernel_size=(1, 2),
    ):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=conv_kernel_size),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=pool_kernel_size, stride=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=conv_kernel_size),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=pool_kernel_size, stride=2),
        )
        self.fc = nn.Sequential(
            nn.Linear(dim_hidden, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out
