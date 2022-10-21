import copy

import ray
import torch
import torch.nn as nn

from blades.datasets.fldataset import FLDataset


@ray.remote
class _RayActor(object):
    """Ray Actor."""

    def __init__(self, dataset: FLDataset, *args, **kwargs):
        """
        Args:
            aggregator (callable): A callable which takes a list of tensors and
            returns an aggregated tensor.
            log_interval (int): Control the frequency of logging training
             batches
            metrics (dict): dict of metric names and their functions
            use_cuda (bool): Use cuda or not
            debug (bool):
        """
        self.dataset = dataset

    def set_global_model(
        self, model: nn.Module, opt: type(torch.optim.Optimizer), lr: float
    ) -> None:
        r"""Deep copy the given global_model to the client.

        Args:
            model: a Torch global_model for current client.
            opt: client optimizer
            lr:  local learning rate
        """
        self.model = copy.deepcopy(model)
        self.optimizer = opt(self.model.parameters(), lr=lr)

    def set_lr(self, lr: float) -> None:
        r"""change the learning rate of the client optimizer.

        Args:
            lr (float): target learning rate.
        """
        for g in self.optimizer.param_groups:
            g["lr"] = lr

    def local_training(self, clients, global_model, local_round, lr, **kwargs):
        """A proxy method that provides local training for a set of clients.

        Args:
            clients: a list of clients.
            global_model: the global global_model from server.
            local_round: number of local SGD.
            lr: client learning rate.
            **kwargs:

        Returns: a list of ``clients``.
        """
        for client in clients:
            global_state_dict = copy.deepcopy(global_model.state_dict())
            self.model.load_state_dict(global_state_dict)
            self.set_lr(lr)

            client.set_global_model_ref(self.model)
            local_dataset = self.dataset.get_train_loader(client.id())

            client.train_global_model(
                train_set=local_dataset, num_batches=local_round, opt=self.optimizer
            )
            client.train_personal_model(
                train_set=local_dataset,
                num_batches=local_round,
                global_state=global_state_dict,
            )

        return clients

    def evaluate(self, clients, global_model, round_number, batch_size, metrics):
        update = []
        self.model.load_state_dict(global_model.state_dict())
        for i in range(len(clients)):
            clients[i].set_global_model_ref(self.model)
            data = self.dataset.get_all_test_data(clients[i].id())
            result = clients[i].evaluate(
                round_number=round_number,
                test_set=data,
                batch_size=batch_size,
                metrics=metrics,
            )
            update.append(result)
        return update
