from torch.utils.data import Dataset


class CustomTensorDataset(Dataset):
    def __init__(self, data_X, data_y, transform_list=None):
        tensors = (data_X, data_y)
        self.tensors = tensors
        self.transforms = transform_list

    def __getitem__(self, index):
        x = self.tensors[0][index]

        if self.transforms:
            x = self.transforms(x)

        y = self.tensors[1][index]

        return x, y

    def __len__(self):
        return self.tensors[1].size(0)
