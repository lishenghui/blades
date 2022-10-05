import torch.nn as nn

from blades.models.backbones.cctnets import cct_7_3x1_32_c100


class CCTNet100(nn.Module):
    def __init__(self):
        super(CCTNet100, self).__init__()
        self.model = cct_7_3x1_32_c100()

    def forward(self, x):
        return self.model(x)


def create_model():
    return CCTNet100(), nn.modules.loss.CrossEntropyLoss()
