import unittest
import torch
from fllib.datasets.catalog import DatasetCatalog, make_dataset
from fllib.datasets import FLDataset


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
        )


class TestDataset(unittest.TestCase):
    # Class variables defining test configurations
    SUBSETS_TO_TEST = [
        {"subset_ids": ["0", "1"], "expected_num_clients": 2},
        {"subset_ids": ["2"], "expected_num_clients": 1},
        {"subset_ids": ["3", "4"], "expected_num_clients": 2},
        {"subset_ids": [], "expected_num_clients": 0},
    ]
    SPLITS_TO_TEST = [
        {"n": 2, "expected_num_subsets": 2, "expected_num_clients_per_subset": 5},
        {"n": 5, "expected_num_subsets": 5, "expected_num_clients_per_subset": 2},
    ]

    @classmethod
    def setUpClass(cls):
        cls.dataset = make_dataset("mnist", num_clients=10)
        cls.maxDiff = None

    @classmethod
    def tearDownClass(cls):
        # Cleanup any resources here
        pass

    def test_custom_dataset(self):
        DatasetCatalog.register_custom_dataset("simple", SimpleDataset)
        dataset = DatasetCatalog.get_dataset({"custom_dataset": "simple"})
        self.assertIsInstance(dataset, SimpleDataset)

    def test_subsets(self):
        # Test dataset subsets
        for subset_config in self.SUBSETS_TO_TEST:
            subset_ids = subset_config["subset_ids"]
            expected_num_clients = subset_config["expected_num_clients"]

            with self.subTest(subset_ids=subset_ids):
                subset = self.dataset.subset(subset_ids)

                self.assertIsInstance(subset, FLDataset)
                self.assertEqual(
                    len(subset.client_ids),
                    expected_num_clients,
                    f"Subset {subset_ids} should have {expected_num_clients} clients",
                )

    def test_splits(self):
        # Test dataset splits
        for split_config in self.SPLITS_TO_TEST:
            n = split_config["n"]
            expected_num_subsets = split_config["expected_num_subsets"]
            expected_num_clients_per_subset = split_config[
                "expected_num_clients_per_subset"
            ]

            with self.subTest(n=n):
                subsets = self.dataset.split(n)
                # breakpoint()
                self.assertEqual(
                    len(subsets),
                    expected_num_subsets,
                    f"{n}-way split should generate {expected_num_subsets} subsets",
                )
                for subset in subsets:
                    self.assertIsInstance(subset, FLDataset)
                    self.assertEqual(
                        len(subset.client_ids),
                        expected_num_clients_per_subset,
                        f"Each subset in a {n}-way split should have "
                        f"{expected_num_clients_per_subset} clients",
                    )

    def test_pickle(self):
        # Test dataset pickling
        import pickle

        pickled_dataset = pickle.dumps(self.dataset)
        unpickled_dataset = pickle.loads(pickled_dataset)

        self.assertIsInstance(unpickled_dataset, FLDataset)
        self.assertEqual(
            len(unpickled_dataset.client_ids),
            len(self.dataset.client_ids),
            "Unpickled dataset should have the same number of clients",
        )
