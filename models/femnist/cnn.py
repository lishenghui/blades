import torch
import torch.nn as nn
import torch.nn.functional as F


class HLoss(nn.Module):
    def __init__(self):
        super(HLoss, self).__init__()
    
    def forward(self, x):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = -1.0 * b.sum()
        return b


class Net(nn.Module):
    def __init__(self, num_classes):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Sequential(nn.Conv2d(1, 32, 5, 1, 1), torch.nn.ReLU())
        self.conv2 = torch.nn.Sequential(nn.Conv2d(32, 64, 5, 1, 1), torch.nn.ReLU())
        self.fc1 = torch.nn.Sequential(nn.Linear(9216, 128), torch.nn.ReLU())
        self.fc2 = torch.nn.Sequential(nn.Linear(128, num_classes))
    
    #     self.fc3 = torch.nn.Sequential(nn.Linear(28 * 28, 128, torch.nn.ReLU()))
    #     self.fc4 = torch.nn.Sequential(nn.Linear(128, num_classes))
    
    # def forward(self, x):
    #     x = torch.flatten(x, 1)
    #     x = self.fc3(x)
    #     x = self.fc4(x)
    #     output = F.log_softmax(x, dim=1)
    #     return output
    
    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        x = self.conv1(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
    
    def test_model(self, test_loader, device):
        self.eval()
        test_loss = 0
        test_entropy = 0
        correct = 0
        test_size = len(test_loader.dataset)
        centropy = HLoss()
        with torch.no_grad():
            for input, target in test_loader:
                input, target = input.to(device), target.to(device)
                output = self(input)
                test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
                test_entropy += centropy(output).item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        
        test_loss /= len(test_loader.dataset)
        test_entropy /= len(test_loader.dataset)
        # print(test_loss)
        return {
            "loss": round(test_loss, 3),
            "entropy": round(test_entropy, 3),
            "accuracy": round(100. * correct / test_size, 3),
            "size": test_size
        }
