import logging
import types
from typing import Optional

import torch

from .mean import _BaseAggregator, _BaseAsyncAggregator

debug_logger = logging.getLogger("debug")


class Centeredclipping(_BaseAggregator):
    r"""A robust aggregator from paper `Learning from History for Byzantine
    Robust Optimization.

    <http://proceedings.mlr.press/v139/karimireddy21a.html>`_. It iteratively
    clips the updates around the center while updating the center accordingly.

    Args:
        tau (float): The threshold of clipping. Default 10.0
        n_iter (int): The number of clipping iterations. Default 5
    """

    def __init__(self, tau: Optional[float] = 10.0, n_iter: Optional[int] = 5):
        self.tau = tau
        self.n_iter = n_iter
        super(Centeredclipping, self).__init__()
        self.momentum = None

    def clip(self, v):
        v_norm = torch.norm(v)
        scale = min(1, self.tau / v_norm)
        return v * scale

    def __call__(self, clients):
        updates = list(map(lambda w: w.get_update(), clients))
        if self.momentum is None:
            self.momentum = torch.zeros_like(updates[0])

        for _ in range(self.n_iter):
            self.momentum = (
                sum(self.clip(v - self.momentum) for v in updates) / len(updates)
                + self.momentum
            )

        return torch.clone(self.momentum).detach()

    def __str__(self):
        return "Clipping (tau={}, n_iter={})".format(self.tau, self.n_iter)


class _AnchorClipping(Centeredclipping):
    def __init__(self, node, weights, opt, model, tau, n_iter=1):
        super(_AnchorClipping, self).__init__(tau, n_iter)
        self._anchor_buffer = self._vectorize_model(model)
        self.opt = self._wrap_step(opt, model)
        assert n_iter == 1
        assert len(weights.shape) == 1
        self.node = node
        self.weights = weights

    def _vectorize_model(self, model):
        """"""
        state_dict = model.state_dict()
        return torch.cat([state_dict[k].data.view(-1) for k in state_dict])

    def _wrap_step(self, opt: torch.optim.Optimizer, model: torch.nn.Module):
        """Wrap the step function of opt to track the change."""
        debug_logger.info("Wrap the step function of opt")

        if hasattr(opt, "_core_step") or hasattr(opt, "anchorclipping"):
            raise NotImplementedError

        # Cache the old class
        opt._core_step = types.MethodType(opt.__class__.step, opt)
        opt.anchorclipping = self

        # Update the anchor vector y every time the `step` is called.
        def anchor_clipping_step(self, closure=None):
            state_dict = model.state_dict()
            flattened = torch.cat([state_dict[k].data.view(-1) for k in state_dict])
            self._core_step(closure=closure)
            after_state_dict = model.state_dict()
            after_flattened = torch.cat(
                [state_dict[k].data.view(-1) for k in after_state_dict]
            )
            # debug_logger.info((after_flattened - flattened)[:5])
            self.anchorclipping._anchor_buffer.add_(after_flattened - flattened)

        opt.step = types.MethodType(anchor_clipping_step, opt)

    def __call__(self, inputs):
        assert len(inputs) == 1 + len(self.node.edges)
        clipped = self._anchor_buffer + self.clip(inputs[0] - self._anchor_buffer)
        s = self.weights[self.node.index] * clipped
        for e, inp in zip(self.node.edges, inputs[1:]):
            theothernode = e.theother(self.node)
            clipped = self._anchor_buffer + self.clip(inp - self._anchor_buffer)
            s += self.weights[theothernode.index] * clipped
        return s

    def __str__(self):
        return "_AnchorClipping(tau={}, n_iter={})".format(self.tau, self.n_iter)


class _AsyncCenteredClipping(_BaseAsyncAggregator):
    """Comparing to Clipping, _AsyncCenteredClipping does not average the
    clipped gradient but use fraction $1 / n$ where `n` is the number of total
    gradients."""

    def __init__(self, tau, n_iter=1):
        self.tau = tau
        self.n_iter = n_iter
        super(_AsyncCenteredClipping, self).__init__()

        self.momentum = 0

    def clip(self, v):
        v_norm = torch.norm(v)
        scale = min(1, self.tau / v_norm)
        return v * scale

    def __call__(self, inputs):
        n = len(inputs)
        filtered = list(filter(lambda x: x is not None, inputs))

        for _ in range(self.n_iter):
            self.momentum = (
                sum(self.clip(v - self.momentum) for v in filtered) / n + self.momentum
            )

        return torch.clone(self.momentum).detach()

    def __str__(self):
        return "_AsyncCenteredClipping (tau={}, n_iter={})".format(
            self.tau, self.n_iter
        )
