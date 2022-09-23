# import pytest
import torch

from blades.aggregators import Mean, Median, Trimmedmean

# from blades.aggregators import (Autogm, Clippedclustering, Clustering, Dnc,
#                                 Fltrust, Geomed, Mean, Median, Multikrum,
#                                 Signguard, Trimmedmean)

test_data = torch.Tensor([
    [1, 2, 3],
    [-1, 4, -1],
    [2., 2, 3],
    [3., 1., 3],
])


def test_median():
    median = Median()
    assert torch.equal(
        median(test_data),
        torch.Tensor([1.5, 2.0, 3.0000]),
    )


def test_mean():
    mean = Mean()
    assert torch.equal(
        mean(test_data),
        torch.Tensor([1.2500, 2.2500, 2.0000]),
    )


def test_trimmedmean():
    tm = Trimmedmean(1)
    assert torch.equal(tm(test_data), torch.Tensor([1.5000, 2.0000, 3.0000]))
