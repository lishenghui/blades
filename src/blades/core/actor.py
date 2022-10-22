from typing import Dict, List, TypeVar, Optional

import ray
import torch
import torch.nn as nn
from blades.clients.client import BladesClient
from blades.datasets.fldataset import FLDataset
from blades.utils.torch_utils import vector_to_parameters

# from blades.utils.torch_utils import parameters_to_vector, vector_to_parameters
from torch.optim import Optimizer
import copy
from blades.utils.utils import set_random_seed
import random
import numpy as np

T = TypeVar("T", bound="Optimizer")


@ray.remote
class Actor(object):
    """Ray Actor."""

    def __init__(
        self,
        dataset: FLDataset,
        model: nn.Module,
        opt_cls: T,
        opt_kws: Dict,
        mem_meta_info: tuple = None,
        buffer_blocks: List[int] = None,
        seed: Optional[int] = 0,
    ):
        """_summary_

        Args:
            dataset (FLDataset): _description_
            id (int): _description_
            mem_meta_info (tuple): _description_
            model (nn.Module): _description_
            opt (torch.optim.Optimizer): _description_
            lr (float): _description_
        #
        """
        set_random_seed(seed)
        self.dataset = dataset
        self.model = model
        self.buffer_blocks = buffer_blocks
        self.optimizer = opt_cls(self.model.parameters(), **opt_kws)
        if mem_meta_info:
            self.shared_memory = mem_meta_info[0](*mem_meta_info[1])
        self.random_states = {}

    def cache_random_state(self) -> None:
        # This function should be used for reproducibility
        if ray.get_gpu_ids() != []:
            self.random_states["torch_cuda"] = torch.cuda.get_rng_state()
        self.random_states["torch"] = torch.get_rng_state()
        self.random_states["numpy"] = np.random.get_state()
        self.random_states["python"] = random.getstate()

    def restore_random_state(self) -> None:
        # This function should be used for reproducibility
        if ray.get_gpu_ids() != []:
            torch.cuda.set_rng_state(self.random_states["torch_cuda"])
        torch.set_rng_state(self.random_states["torch"])
        np.random.set_state(self.random_states["numpy"])
        random.setstate(self.random_states["python"])

    def init(self):
        return True

    def set_lr(self, lr: float) -> None:
        r"""change the learning rate of the client optimizer.

        Args:
            lr (float): target learning rate.
        """
        for g in self.optimizer.param_groups:
            g["lr"] = lr

    def local_train(
        self,
        clients: List[BladesClient],
        *,
        num_rounds: int = 1,
        global_model: nn.Module = None,
    ) -> List:
        """A proxy method that provides local training for a set of clients.

        Args:
            clients (List): a list of clients.
            num_rounds (int, optional): number of local steps. Defaults to 1.
            global_model (nn.Module, optional): the global global_model from server. \
                                                Defaults to None.

        Returns:
            List: a list of the given clients.
        """
        # self.cache_random_state()
        if not global_model:
            model_vec = copy.deepcopy(self.shared_memory[0])
        for client in clients:
            if global_model:
                self.model.load_state_dict(copy.deepcopy(global_model.state_dict()))
            else:
                vector_to_parameters(copy.deepcopy(model_vec), self.model.parameters())
            client.set_global_model_ref(self.model)
            local_dataset = self.dataset.get_train_loader(client.id())
            client.train_global_model(
                train_set=local_dataset, num_batches=num_rounds, opt=self.optimizer
            )
            client.train_personal_model(
                train_set=local_dataset,
                num_batches=num_rounds,
                global_state=self.model.state_dict(),
            )

        update = torch.stack(list(map(lambda w: w.get_update(), clients)))
        self.shared_memory[
            self.buffer_blocks,
        ] = update
        # self.restore_random_state()
        return clients

    def evaluate(
        self,
        clients,
        global_model: torch.nn.Module = None,
        round_number: int = None,
        batch_size: int = 128,
        metrics=None,
    ):
        update = []
        if not global_model:
            vector_to_parameters(self.shared_memory[0], self.model.parameters())

        self.model.eval()
        for client in clients:
            client.set_global_model_ref(self.model)
            data = self.dataset.get_all_test_data(client.id())
            result = client.evaluate(
                round_number=round_number,
                test_set=data,
                batch_size=batch_size,
                metrics=metrics,
            )
            update.append(result)
        return update
