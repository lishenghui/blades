import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import matthews_corrcoef


class Net(nn.Module):
    def __init__(self, input_size, hidden_size=512):
        super(Net, self).__init__()
        self.drop = nn.Dropout(p=0.0)
        self.layer1 = torch.nn.Linear(input_size, hidden_size)
        self.layer2 = torch.nn.Linear(hidden_size, hidden_size)
        self.layer3 = torch.nn.Linear(hidden_size, 2)
        # self.bn = nn.BatchNorm1d(hidden_size)
    
    def forward(self, x):
        x = self.layer1(x)
        x = nn.ReLU()(self.drop(x))
        x = self.layer2(x)
        x = nn.ReLU()(self.drop(x))
        out = self.layer3(x)
        out = F.log_softmax(out, dim=1)
        return out
    
    def test_model(self, test_loader, device):
        self.eval()
        test_loss = 0
        correct = 0
        test_size = len(test_loader.dataset)
        
        predictions = torch.empty((1, 1)).to(device)
        true_labels = torch.empty((1, 1)).to(device)
        with torch.no_grad():
            for input, target in test_loader:
                input, target = input.to(device), target.to(device)
                output = self(input)
                test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                
                predictions = torch.vstack([predictions, pred])
                true_labels = torch.vstack([true_labels, target.view_as(pred)])
        test_loss /= len(test_loader.dataset)
        
        predictions = predictions[1:].int().cpu().numpy().flatten()
        true_labels = true_labels[1:].int().cpu().numpy().flatten()
        # print(predictions, true_labels)
        mcc = matthews_corrcoef(predictions, true_labels)
        # print(mcc)
        return {
            "loss": test_loss,
            "accuracy": mcc,
            "size": test_size
        }
