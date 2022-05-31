import os
import torch
from torch import nn
import torch.nn.functional as F


class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()
        self.flatten = nn.Flatten()
        self.layer1 = nn.Linear(28*28, 64)
        self.layer2 = nn.Linear(64, 128)
        self.layer3 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.log_softmax(self.layer3(x), dim=1)
        return x


def create_model():
    return DNN(), nn.modules.loss.CrossEntropyLoss()
