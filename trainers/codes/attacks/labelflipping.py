import os

from torchvision import datasets

from ..simulators.worker import ByzantineWorker


class LabelflippingMNIST(datasets.MNIST):
    def __getitem__(self, index):
        img, target = super(LabelflippingMNIST, self).__getitem__(index)
        target = 9 - target
        return img, target

    @property
    def raw_folder(self):
        return os.path.join(self.root, "MNIST", "raw")

    @property
    def processed_folder(self):
        return os.path.join(self.root, "MNIST", "processed")


class LabelflippingCIFAR10(datasets.CIFAR10):
    def __getitem__(self, index):
        img, target = super(LabelflippingCIFAR10, self).__getitem__(index)
        target = 9 - target
        return img, target


class LableFlippingWorker(ByzantineWorker):
    def __init__(self, revertible_label_transformer, *args, **kwargs):
        """
        Args:
            revertible_label_transformer (callable):
                E.g. lambda label: 9 - label
        """
        super().__init__(*args, **kwargs)
        self.revertible_label_transformer = revertible_label_transformer

    def train_epoch_start(self) -> None:
        super().train_epoch_start()
        self.running["train_loader_iterator"].__next__ = self._wrap_iterator(
            self.running["train_loader_iterator"].__next__
        )

    def _wrap_iterator(self, func):
        def wrapper():
            data, target = func()
            return data, self.revertible_label_transformer(target)

        return wrapper

    def _wrap_metric(self, func):
        def wrapper(output, target):
            return func(output, self.revertible_label_transformer(target))

        return wrapper

    def add_metric(self, name, callback):
        if name in self.metrics or name in ["loss", "length"]:
            raise KeyError(f"Metrics ({name}) already added.")

        self.metrics[name] = self._wrap_metric(callback)

    def __str__(self) -> str:
        return "LableFlippingWorker"