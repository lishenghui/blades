import unittest

import torch

from fllib.aggregators import Mean, Median, Trimmedmean, GeoMed, DnC


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
        self.mean = Mean()
        self.median = Median()
        self.trimmedmean = Trimmedmean(1)
        self.geomed = GeoMed(eps=1e-8, maxiter=1000, ftol=1e-22)
        self.dnc = DnC(num_byzantine=1, sub_dim=2, num_iters=1)

    def test_mean(self):
        expected_output = torch.Tensor([1.2500, 2.2500, 2.0000])
        self.assertTrue(torch.allclose(self.mean(self.input_data), expected_output))

    def test_median(self):
        expected_output = torch.Tensor([1.5, 2.0, 3.0000])
        self.assertTrue(torch.allclose(self.median(self.input_data), expected_output))

    def test_trimmedmean(self):
        expected_output = torch.Tensor([1.5000, 2.0000, 3.0000])
        self.assertTrue(
            torch.allclose(self.trimmedmean(self.input_data), expected_output)
        )

    def test_geomed(self):
        def compute_condition_vector(inputs, median):
            # Compute the condition vector using the characterization of the geometric
            # median, see this link: https://en.wikipedia.org/wiki/Geometric_median.
            return torch.vstack(
                [(input - median) / torch.norm(input - median) for input in inputs]
            ).sum(dim=0)

        output = self.geomed(self.input_data)
        condi = compute_condition_vector(self.input_data, output)
        self.assertTrue(torch.allclose(condi, torch.zeros_like(condi), atol=1e-3))

    def test_dnc(self):
        output = self.dnc(self.input_data)
        # TODO: check the correctness of the expected output
        expected_output = torch.Tensor([2.0000, 1.6667, 3.0000])
        self.assertTrue(torch.allclose(output, expected_output, atol=1e-4))


if __name__ == "__main__":
    unittest.main()
