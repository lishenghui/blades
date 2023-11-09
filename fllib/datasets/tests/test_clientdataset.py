import pickle
import unittest

import torch
from torch.utils.data import Dataset

from fllib.datasets import ClientDataset


class SimpleDataset(Dataset):
    def __init__(self):
        # Assuming self.data is a list of torch Tensors
        self.data = [torch.tensor([i], dtype=torch.float32) for i in range(10)]

    def __len__(self):
        # Returns the size of the dataset
        return len(self.data)

    def __getitem__(self, index):
        # Retrieves the data at the given index
        return self.data[index]


class TestClientDataset(unittest.TestCase):
    def setUp(self):
        # Create a small Ray dataset for testing purposes
        self.dataset = SimpleDataset()
        self.uid = "test_uid"
        self.batch_size = 2  # Small batch size for testing

        # Instantiate the ClientDataset
        self.client_dataset = ClientDataset(
            uid=self.uid, train_set=self.dataset, train_batch_size=self.batch_size
        )

    def test_get_next_batch(self):
        # Retrieve the first batch
        first_batch = self.client_dataset.get_next_train_batch()
        print(first_batch)
        self.assertIsNotNone(first_batch, "The first batch should not be None")
        self.assertEqual(
            first_batch.shape[0],
            self.batch_size,
            f"The batch size should be {self.batch_size}",
        )

        # Retrieve all batches and ensure they are returned correctly
        all_batches = [first_batch]
        for _ in range(1, len(self.dataset) // self.batch_size):
            batch = self.client_dataset.get_next_train_batch()
            self.assertIsNotNone(batch, "Subsequent batches should not be None")
            self.assertEqual(
                len(batch),
                self.batch_size,
                f"The batch size should be {self.batch_size}",
            )
            all_batches.append(batch)

        # Check if all data points were iterated over
        all_data = torch.cat(all_batches)
        expected_data = torch.cat(self.dataset.data).view(-1, 1)[: len(all_data)]
        torch.testing.assert_allclose(all_data, expected_data)

    def test_pickle_dataset(self):
        # Pickle the client dataset
        pickled_dataset = pickle.dumps(self.client_dataset)

        # Unpickle the client dataset
        unpickled_dataset: ClientDataset = pickle.loads(pickled_dataset)

        # Check if the unpickled dataset is the same type
        self.assertIsInstance(
            unpickled_dataset,
            ClientDataset,
            "Unpickled object is not a ClientDataset instance.",
        )

        # Check if the unpickled dataset has the same uid
        self.assertEqual(
            unpickled_dataset.uid,
            self.client_dataset.uid,
            "UID of unpickled dataset does not match.",
        )

        # Check if the unpickled dataset can generate batches
        try:
            batch = unpickled_dataset.get_next_train_batch()
            self.assertIsNotNone(
                batch, "The batch from unpickled dataset should not be None."
            )
        except Exception as e:
            self.fail(
                f"Getting next batch from unpickled dataset failed with exception: {e}"
            )


if __name__ == "__main__":
    unittest.main()
