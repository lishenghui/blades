from typing import List, Union

import torch

from blades.clients.client import BladesClient


class _BaseAggregator(object):
    """Base class of aggregators.

    Args:
        dist_communicator (object): A link object which can broadcast / gather, etc.
    """

    def __init__(self, *args, **kwargs):
        pass
        # log("Init aggregators: " + self.__str__())
        # log_dict({"Aggregator": self.__str__(), "Type": "Setup"})

    def _get_updates(
        self, inputs: Union[List[BladesClient], List[torch.Tensor], torch.Tensor]
    ):
        if all(isinstance(element, BladesClient) for element in inputs):
            updates = torch.stack(list(map(lambda w: w.get_update(), inputs)))
        elif isinstance(inputs, List) and all(
            isinstance(element, torch.Tensor) for element in inputs
        ):
            updates = torch.stack(inputs, dim=0)
        else:
            updates = inputs
        return updates

    def __call__(self, inputs):
        """Aggregate the inputs and update in-place.

        Args:
            inputs (list): A list of tensors to be aggregated.

        Raises:
            NotImplementedError:
        """
        raise NotImplementedError


class Mean(_BaseAggregator):
    r"""Computes the ``sample mean`` over the updates from all give clients."""

    def __int__(self):
        super(Mean, self).__init__()

    def __call__(
        self, inputs: Union[List[BladesClient], List[torch.Tensor], torch.Tensor]
    ):
        updates = self._get_updates(inputs)
        values = updates.mean(dim=0)
        # breakpoint()
        return values

    def __str__(self):
        return "Mean"
