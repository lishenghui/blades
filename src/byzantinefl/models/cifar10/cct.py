import torch.nn as nn

from .cctnets import cct_2_3x2_32


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.mdoel = cct_2_3x2_32()
    
    def forward(self, x):
        return self.mdoel(x)
