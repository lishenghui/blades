import torch
from torch.utils.data import DataLoader


class ClientDataset:
    def __init__(
        self,
        uid: str,
        train_set: torch.utils.data.Dataset = None,
        test_set: torch.utils.data.Dataset = None,
        train_batch_size: int = 32,
        test_batch_size: int = 32,
        num_workers: int = 0,
    ):
        self._uid = uid
        self._train_set = train_set
        self._test_set = test_set
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers
        self.train_loader = None  # DataLoader for training will be created when needed
        self.test_loader = None  # DataLoader for testing will be created when needed

    @property
    def uid(self):
        """Returns the unique identifier of the client dataset."""
        return self._uid

    @property
    def train_set_size(self):
        if self._train_set is None:
            return 0
        return len(self._train_set)

    @property
    def train_set(self):
        return self._train_set

    @property
    def test_set(self):
        return self._test_set

    @property
    def test_set_size(self):
        if self._test_set is None:
            return 0
        return len(self._test_set)

    def _create_train_loader(self):
        # Creates a new DataLoader for training when called
        self.train_loader = iter(
            DataLoader(
                self._train_set,
                batch_size=self.train_batch_size,
                num_workers=self.num_workers,
            )
        )

    def _create_test_loader(self):
        # Creates a new DataLoader for testing when called
        self.test_loader = DataLoader(
            self._test_set,
            batch_size=self.test_batch_size,
            num_workers=self.num_workers,
        )
        # )

    def get_next_train_batch(self):
        if self.train_loader is None:
            self._create_train_loader()

        try:
            # Return the next train batch
            return next(self.train_loader)
        except StopIteration:
            # Re-create the train DataLoader if the previous loader is exhausted
            self._create_train_loader()
            return next(self.train_loader)

    def get_train_loader(self):
        if self.train_loader is None:
            self._create_train_loader()
        return self.train_loader

    def get_test_loader(self):
        if self.test_loader is None:
            self._create_test_loader()
        return self.test_loader

    def __getstate__(self):
        # Return the state of the object for serialization
        # Exclude both DataLoaders from serialization
        state = self.__dict__.copy()
        state["train_loader"] = None
        state["test_loader"] = None
        return state

    def __setstate__(self, state):
        # Restore the state of the object after serialization
        # Ignore the DataLoaders, they will be created when needed
        self.__dict__.update(state)
        self.train_loader = None  # Train DataLoader will be created dynamically
        self.test_loader = None  # Test DataLoader will be created dynamically

    def __repr__(self):
        return (
            f"Client ID: {self.uid} \n"
            f"[Train set size: {self.train_set_size} \n"
            f"Test set size: {self.test_set_size}]"
        )
