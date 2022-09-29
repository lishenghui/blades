import torch.nn as nn

from blades.models.backbones.cctnets import cct_7_3x1_32_c100


class CCTNet(nn.Module):
    def __init__(self):
        super(CCTNet, self).__init__()
        self.model = cct_7_3x1_32_c100()

    def forward(self, x):
        return self.model(x)


def create_model():
    return CCTNet(), nn.modules.loss.CrossEntropyLoss()
