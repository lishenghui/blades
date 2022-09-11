import torch.nn as nn

from .cctnets import cct_2_3x2_32


class CCTNet(nn.Module):
    def __init__(self):
        super(CCTNet, self).__init__()
        self.mdoel = cct_2_3x2_32()
    
    def forward(self, x):
        return self.mdoel(x)


def create_model():
    return CCTNet(), nn.modules.loss.CrossEntropyLoss()
