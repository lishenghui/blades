import ray

from blades.datasets.dataset import FLDataset
import torch
import torch.nn as nn
import copy


@ray.remote
class _RayActor(object):
    """Ray Actor."""

    def __init__(self, dataset: object, *args, **kwargs):
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
        traindls, testdls = dataset.get_dls()
        self.dataset = FLDataset(traindls, testdls)

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

    def load_global_para(self, model):
        state_dict = model.state_dict()
        self.model.load_state_dict(state_dict)

    def set_lr(self, lr: float) -> None:
        r"""change the learning rate of the client optimizer.

        Args:
            lr (float): target learning rate.
        """

        for g in self.optimizer.param_groups:
            g["lr"] = lr

    def local_training(self, clients, model, local_round, lr, *args, **kwargs):
        if "dp" in kwargs and kwargs["dp"] is True:
            assert "clip_threshold" in kwargs and "noise_factor" in kwargs
            dp = True
            clip_threshold = kwargs["clip_threshold"]
            noise_factor = kwargs["noise_factor"]
        else:
            dp = False
            clip_threshold = 0
            noise_factor = 0
        update = []

        for i in range(len(clients)):
            self.load_global_para(model)
            self.set_lr(lr)

            self.model.train()
            clients[i].set_model_ref(self.model)
            clients[i].on_train_round_begin(self.model)
            data = self.dataset.get_train_data(clients[i].id(), local_round)

            clients[i].local_training(data_batches=data, opt=self.optimizer)

            clients[i].on_train_round_end(
                dp=dp,
                clip_threshold=clip_threshold,
                noise_factor=noise_factor,
            )
            update.append(clients[i].get_update())
        return clients

    def evaluate(self, clients, model, round_number, batch_size, metrics):
        update = []
        self.load_global_para(model)
        for i in range(len(clients)):
            # clients[i].set_para(global_model)
            clients[i].set_model_ref(self.model)
            data = self.dataset.get_all_test_data(clients[i].id())
            result = clients[i].evaluate(
                round_number=round_number,
                test_set=data,
                batch_size=batch_size,
                metrics=metrics,
            )
            update.append(result)
        return update
