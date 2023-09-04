from typing import Optional, List

import torch


class Centeredclipping(object):
    r"""A robust aggregator from paper `Learning from History for Byzantine
    Robust Optimization.

    <http://proceedings.mlr.press/v139/karimireddy21a.html>`_. It iteratively
    clips the updates around the center while updating the center accordingly.

    Args:
        tau (float): The threshold of clipping. Default 10.0
        n_iter (int): The number of clipping iterations. Default 5
    """

    def __init__(self, tau: Optional[float] = 5.0, n_iter: Optional[int] = 5):
        self.tau = tau
        self.n_iter = n_iter
        self.momentum = None

    def clip(self, v):
        v_norm = torch.norm(v)
        scale = min(1, self.tau / v_norm)
        return v * scale

    def __call__(self, inputs: List[torch.Tensor]):
        if self.momentum is None:
            self.momentum = torch.zeros_like(inputs[0])

        for _ in range(self.n_iter):
            self.momentum = (
                sum(self.clip(v - self.momentum) for v in inputs) / len(inputs)
                + self.momentum
            )

        return torch.clone(self.momentum).detach()
