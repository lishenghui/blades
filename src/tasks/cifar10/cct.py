import torch
import torch.nn as nn
import torch.nn.functional as F
from .cctnets import cct_2_3x2_32

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.mdoel = cct_2_3x2_32()
    
    def forward(self, x):
        return self.mdoel(x)
    
    # def test_model(self, test_loader, device):
    #     self.eval()
    #     test_loss = 0
    #     correct = 0
    #     test_size = len(test_loader.dataset)
    #
    #     with torch.no_grad():
    #         for input, target in test_loader:
    #             input, target = input.to(device), target.to(device)
    #             output = self(input)
    #             test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
    #             pred = output.argmax(dim=1, keepdim=True)
    #             correct += pred.eq(target.view_as(pred)).sum().item()
    #
    #     test_loss /= len(test_loader.dataset)
    #
    #     return {
    #         "loss": test_loss,
    #         "accuracy": 100. * correct / test_size,
    #         "size": test_size
    #     }
