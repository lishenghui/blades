import unittest
import uuid

from fllib.datasets.dataset import FLDataset


class _ClientData:
    def __init__(self, data):
        self.data = data
        self.uid = str(uuid.uuid4())


# Generate a mock dataset with unique identifiers
def generate_mock_datasets(num_clients: int, data_size_per_client: int):
    # Generate a ClientData instance for each client dataset with a UID
    return [_ClientData(list(range(data_size_per_client))) for _ in range(num_clients)]


class TestFLDataset(unittest.TestCase):
    def setUp(self):
        self.num_clients = 10
        self.data_size_per_client = 5
        client_datasets = generate_mock_datasets(
            self.num_clients, self.data_size_per_client
        )
        self.fl_dataset = FLDataset(client_datasets)

    def test_length(self):
        self.assertEqual(
            len(self.fl_dataset), self.num_clients, "Length of FLDataset is incorrect"
        )

    def test_split_even(self):
        num_shards = 2
        shards = self.fl_dataset.split(num_shards)
        self.assertEqual(
            len(shards), num_shards, "Split into incorrect number of shards"
        )
        self.assertTrue(
            all(isinstance(shard, FLDataset) for shard in shards),
            "Shards are not instances of FLDataset",
        )

        expected_shard_size = self.num_clients // num_shards
        for shard in shards[
            :-1
        ]:  # All but the last shard should have the expected shard size
            self.assertEqual(len(shard.client_datasets), expected_shard_size)

    def test_split_uneven(self):
        num_shards = 3
        shards = self.fl_dataset.split(num_shards)
        self.assertTrue(
            len(shards) == num_shards or len(shards) == num_shards - 1,
            "Split into incorrect number of shards",
        )
        self.assertTrue(
            all(isinstance(shard, FLDataset) for shard in shards),
            "Shards are not instances of FLDataset",
        )

    def test_pickle(self):
        # Test dataset pickling
        import pickle

        pickled_dataset = pickle.dumps(self.fl_dataset)
        unpickled_dataset = pickle.loads(pickled_dataset)

        self.assertIsInstance(unpickled_dataset, FLDataset)
        self.assertEqual(
            len(unpickled_dataset.client_ids),
            len(self.fl_dataset.client_ids),
            "Unpickled dataset should have the same number of clients",
        )

    # def test_to_torch():
    #     torch_dataset = self.fl_dataset.to_torch()
