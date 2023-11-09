import unittest

import torch

from fllib.aggregators import Clippedclustering


class TestAggregators(unittest.TestCase):
    def setUp(self):
        raw_input = torch.Tensor(
            [
                [1, 2, 3],
                [-1, 4, -1],
                [2.0, 2, 3],
                [3.0, 1.0, 3],
            ]
        )
        self.input_data = [
            torch.squeeze(tensor) for tensor in torch.split(raw_input, 1)
        ]
        self.clippedclustering = Clippedclustering()

    def test_clippedclustering(self):
        expected_output = torch.Tensor([1.9596, 1.6532, 2.9596])
        output = self.clippedclustering(self.input_data)
        self.assertTrue(torch.allclose(output, expected_output))
