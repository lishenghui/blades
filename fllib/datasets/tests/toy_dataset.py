import torch
from torch.utils.data import Dataset

from fllib.datasets import FLDataset
from fllib.datasets.splitters import IIDSplitter


class _ToyDataset(Dataset):
    def __init__(self):
        self.features = torch.tensor([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
        self.targets = torch.tensor([1, 0, 0])

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        x = self.features[index]
        y = self.targets[index]
        return x, y


class ToyFLDataset(FLDataset):
    num_clients = 3

    def __init__(self) -> None:
        toy_train_set = _ToyDataset()
        splitter = IIDSplitter(num_clients=self.num_clients)
        client_datasets = splitter.generate_client_datasets(
            toy_train_set,
            toy_train_set,
            train_batch_size=1,
            test_batch_size=1,
        )
        super().__init__(client_datasets)
