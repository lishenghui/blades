import torch
from fllib.datasets.fldataset import FLDataset


class SimpleDataset(FLDataset):
    def __init__(
        self,
        cache_name: str = "",
        iid=True,
        alpha=0.1,
        num_clients=3,
        seed=1,
        train_data=None,
        test_data=None,
        train_bs=1,
        num_classes=2,
    ) -> None:
        # Simple dataset
        features = torch.tensor([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]]).numpy()
        targets = torch.tensor([1.0, 0.0, 0.0]).numpy()
        super().__init__(
            (features, targets),
            (features, targets),
            cache_name=cache_name,
            iid=iid,
            alpha=alpha,
            num_clients=num_clients,
            seed=seed,
            train_data=train_data,
            test_data=test_data,
            train_bs=train_bs,
            is_image=False,
        )
