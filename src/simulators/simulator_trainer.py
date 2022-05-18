import logging
from typing import Union, Callable, Any

import numpy as np
import ray
import torch
from ray.train import Trainer
import time
from threading import Thread
from .server import TorchServer
from .client import TorchClient
from concurrent.futures import ThreadPoolExecutor, as_completed
# from .simulator import DistributedSimulatorBase, ParallelTrainer, DistributedTrainerBase


# def train_function(config):
#     config['client'].set_para(config['model'])
#     config['client'].train_epoch_start()
#     config['trainer'].run(config['client'].local_training, config)
#     # config['client'].local_training(config['local_round'])
#     print('training complete')
#     return config['client'].get_update()
    
class RayTrainer(Trainer):
    def __int__(self, backend="torch", num_workers=1, use_gpu=False):
        super().__init__(backend="torch", num_workers=num_workers, use_gpu=use_gpu)
    
    # def run(self, *args, **kwargs):
    #     return super().run(train_function, **kwargs)
        
class TrainerPool(ThreadPoolExecutor):
    def __init__(self, server, num_worker=4):
        super().__init__(max_workers=num_worker)
        self.ray_trainers = [Trainer(backend="torch", num_workers=1, use_gpu=False) for _ in range(num_worker)]
        [trainer.start() for trainer in self.ray_trainers]
        self.server = server
        self.clients =None
        self._return = None

    def set_clients(self, clients):
        self.clients = clients
        
    def run(self):
        self._return = []
        for client in self.clients:
            update = self.ray_trainer.run(train_function, config= {'client' : client, 'model' : self.server.get_model(), 'local_round' : 50})
            self._return.extend(update)

    def join(self):
        Thread.join(self)
        return self._return

class DistributedSimulatorBase(object):
    """Simulate distributed programs with low memory usage.

    Functionality:
    1. randomness control: numpy, torch, torch-cuda
    2. add workers

    This base class is used by both trainer and evaluator.
    """
    
    def __init__(self, metrics: dict, use_cuda: bool, debug: bool):
        """
        Args:
            metrics (dict): dict of metric names and their functions
            use_cuda (bool): Use cuda or not
            debug (bool):
        """
        self.metrics = metrics
        self.use_cuda = use_cuda
        self.debug = debug
        self.clients = []
        # NOTE: omniscient_callbacks are called before aggregation or gossip
        self.omniscient_callbacks = []
        self.random_states = {}
        
        self.json_logger = logging.getLogger("stats")
        self.debug_logger = logging.getLogger("debug")
        self.debug_logger.info(self.__str__())
    
    def __str__(self):
        return f"DistributedSimulatorBase(metrics={list(self.metrics.keys())},use_cuda={self.use_cuda}, debug={self.debug})"
    
    def add_client(self, client: TorchClient):
        client.add_metrics(self.metrics)
        self.debug_logger.info(f"=> Add worker {client}")
        self.clients.append(client)
    
    def any_call(self, f: Callable[[TorchClient], None]) -> None:
        f(self.clients[0])
    
    def any_get(self, f: Callable[[TorchClient], Any]) -> list:
        return f(self.clients[0])
    
    def cache_random_state(self) -> None:
        if self.use_cuda:
            self.random_states["torch_cuda"] = torch.cuda.get_rng_state()
        self.random_states["torch"] = torch.get_rng_state()
        self.random_states["numpy"] = np.random.get_state()
    
    def restore_random_state(self) -> None:
        if self.use_cuda:
            torch.cuda.set_rng_state(self.random_states["torch_cuda"])
        torch.set_rng_state(self.random_states["torch"])
        np.random.set_state(self.random_states["numpy"])
    
    def register_omniscient_callback(self, callback):
        self.omniscient_callbacks.append(callback)


class DistributedTrainerBase(DistributedSimulatorBase):
    """Base class of all distributed training classes."""
    
    def __init__(
            self,
            max_batches_per_epoch: int,
            log_interval: int,
            metrics: dict,
            use_cuda: bool,
            debug: bool,
    ):
        """
        Args:
            max_batches_per_epoch (int): Set the maximum number of batches in an epoch.
                Usually used for debugging.
            log_interval (int): Control the frequency of logging training batches
            metrics (dict): dict of metric names and their functions
            use_cuda (bool): Use cuda or not
            debug (bool):
        """
        self.log_interval = log_interval
        self.max_batches_per_epoch = max_batches_per_epoch
        super().__init__(metrics, use_cuda, debug)
    
    def __str__(self):
        return (
            "DistributedTrainerBase("
            f"max_batches_per_epoch={self.max_batches_per_epoch}, "
            f"log_interval={self.log_interval}, "
            f"metrics={list(self.metrics.keys())}"
            f"use_cuda={self.use_cuda}, "
            f"debug={self.debug}, "
            ")",
        )


class ParallelTrainer(DistributedTrainerBase):
    """Synchronous and parallel training with specified aggregators."""
    def __init__(
            self,
            server: TorchServer,
            aggregator: Callable[[list], torch.Tensor],
            pre_batch_hooks: list,
            post_batch_hooks: list,
            max_batches_per_epoch: int,
            log_interval: int,
            metrics: dict,
            use_cuda: bool,
            debug: bool,
    ):
        """
        Args:
            aggregator (callable): A callable which takes a list of tensors and returns
                an aggregated tensor.
            max_batches_per_epoch (int): Set the maximum number of batches in an epoch.
                Usually used for debugging.
            log_interval (int): Control the frequency of logging training batches
            metrics (dict): dict of metric names and their functions
            use_cuda (bool): Use cuda or not
            debug (bool):
        """
        self.aggregator = aggregator
        self.server = server
        self.pre_batch_hooks = pre_batch_hooks or []
        self.post_batch_hooks = post_batch_hooks or []
        super().__init__(max_batches_per_epoch, log_interval, metrics, use_cuda, debug)
        # self.trainers = [ClientThread(server) for i in range(4)]
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.ray_trainers = [RayTrainer(backend="torch", num_workers=1, use_gpu=False) for _ in range(4)]
        [trainer.start() for trainer in self.ray_trainers]
        
    def parallel_call(self, f: Callable[[TorchClient], None]) -> None:
        self.cache_random_state()
        _ = [f(worker) for worker in self.clients]
        self.restore_random_state()
    
    def parallel_get(self, f: Callable[[TorchClient], Any]) -> list:
        # results = ray.get([f(worker) for worker in self.workers])
        results = []
        for w in self.clients:
            self.cache_random_state()
            results.append(f(w))
            self.restore_random_state()
        return results
    
    def aggregation_and_update(self, log_var=True):
        # If there are Byzantine workers, ask them to craft attackers based on the updated models.
        for omniscient_attacker_callback in self.omniscient_callbacks:
            omniscient_attacker_callback()
        
        gradients = self.parallel_get(lambda w: w.get_gradient())
        if log_var:
            var = torch.norm(torch.var(torch.vstack(gradients), dim=0, unbiased=False)).item()
            print(var)
        aggregated = self.aggregator(gradients)
        
        # Assume that the model and optimizers are shared among workers.
        self.server.set_gradient(aggregated)
        self.server.apply_gradient()
    
    def aggregation_and_update_fedavg(self):
        # If there are Byzantine workers, ask them to craft attackers based on the updated models.
        for omniscient_attacker_callback in self.omniscient_callbacks:
            omniscient_attacker_callback()
        
        update = self.parallel_get(lambda w: w.get_update())
        
        aggregated = self.aggregator(update)
        
        self.server.apply_update(aggregated)
    
    def train(self, epoch):
        self.debug_logger.info(f"Train epoch {epoch}")
        self.parallel_call(lambda worker: worker.train_epoch_start.remote())
        
        progress = 0
        for batch_idx in range(self.max_batches_per_epoch):
            try:
                self.parallel_call(lambda worker: worker.set_para.remote(self.server.get_model()))
                self._run_pre_batch_hooks(epoch, batch_idx)
                results = self.parallel_call(lambda w: w.compute_gradient.remote())
                # self.aggregation_and_update()
                # If there are Byzantine workers, ask them to craft attackers based on the updated models.
                for omniscient_attacker_callback in self.omniscient_callbacks:
                    omniscient_attacker_callback()
                
                gradients = self.parallel_get(lambda w: w.get_gradient.remote())
                aggregated = self.aggregator(gradients)
                
                # Assume that the model and optimizers are shared among workers.
                self.server.set_gradient(aggregated)
                self.server.apply_gradient()
                
                # progress += results[0]["length"]
                if batch_idx % self.log_interval == 0:
                    # self.log_train(progress, batch_idx, epoch, results)
                    self.log_variance(epoch, gradients)
                self._run_post_batch_hooks(epoch, batch_idx)
            except StopIteration:
                continue
    
    def train_fedavg(self, epoch):
        self.debug_logger.info(f"Train epoch {epoch}")
        update = []

        def local_training(config):
            config['client'].set_para(config['model'])
            config['client'].train_epoch_start()
            config['client'].local_training()
            # config['client'].local_training(config['local_round'])
            print('training complete')
            return config['client'].get_update()
        
        def train_function(client, trainer, model, round):
            client.set_para(model)
            client.train_epoch_start()
            return trainer.run(local_training, config=
                            {'client' : client, 'trainer': trainer, 'model' : self.server.get_model(), 'local_round' : 50})
            
        # all_tasks = [self.executor.submit(train_function, config=
        #                     {'client' : client, 'trainer': trainer, 'model' : self.server.get_model(), 'local_round' : 50})
        #             for client, trainer in zip(self.clients, self.ray_trainers)]
        all_tasks = [self.executor.submit(train_function, client, trainer, self.server.get_model(), 50)
                     for client, trainer in zip(self.clients, self.ray_trainers)]
        
        update = [task.result() for task in as_completed(all_tasks)]
        
        # for trainer, client in zip(self.trainers, self.clients):
        #     trainer.set_clients([client])
        #
        # for trainer in self.trainers:
        #     trainer.start()
        #
        # for trainer in self.trainers:
        #     update.extend(trainer.join())
        
        # update = self.parallel_get(lambda w: w.get_update())
        aggregated = self.aggregator(update)
        
        self.server.apply_update(aggregated)
        
        self.log_variance(epoch, update)
    
    def _run_pre_batch_hooks(self, epoch, batch_idx):
        [f(self, epoch, batch_idx) for f in self.pre_batch_hooks]
    
    def _run_post_batch_hooks(self, epoch, batch_idx):
        [f(self, epoch, batch_idx) for f in self.post_batch_hooks]
    
    def log_variance(self, round, update):
        var_avg = torch.mean(torch.var(torch.vstack(update), dim=0, unbiased=False)).item()
        norm = torch.norm(torch.var(torch.vstack(update), dim=0, unbiased=False)).item()
        avg_norm = torch.mean(torch.var(torch.vstack(update), dim=0, unbiased=False) / (
            torch.mean(torch.vstack(update) ** 2, dim=0))).item()
        r = {
            "_meta": {"type": "variance"},
            "E": round,
            "avg": var_avg,
            "norm": norm,
            "avg_norm": avg_norm,
        }
        # Output to file
        self.json_logger.info(r)
    
    def log_train(self, progress, batch_idx, epoch, results):
        length = sum(res["length"] for res in results)
        
        r = {
            "_meta": {"type": "train"},
            "E": epoch,
            "B": batch_idx,
            "Length": length,
            "Loss": sum(res["loss"] * res["length"] for res in results) / length,
        }
        
        for metric_name in self.metrics:
            r[metric_name] = (
                    sum(res["metrics"][metric_name] * res["length"] for res in results)
                    / length
            )
        
        # Output to console
        total = len(self.clients[0].data_loader.dataset)
        pct = 100 * progress / total
        self.debug_logger.info(
            f"[E{r['E']:2}B{r['B']:<3}| {progress:6}/{total} ({pct:3.0f}%) ] Loss: {r['Loss']:.4f} "
            + " ".join(name + "=" + "{:>8.4f}".format(r[name]) for name in self.metrics)
        )
        
        # Output to file
        self.json_logger.info(r)


class DistributedEvaluator(DistributedSimulatorBase):
    def __init__(
            self,
            model: torch.nn.Module,
            data_loader: torch.utils.data.DataLoader,
            loss_func: torch.nn.modules.loss._Loss,
            device: Union[torch.device, str],
            metrics: dict,
            use_cuda: bool,
            debug: bool,
    ):
        super().__init__(metrics, use_cuda, debug)
        self.model = model
        self.data_loader = data_loader
        self.loss_func = loss_func
        self.device = device
    
    def __str__(self):
        return (
            "DistributedEvaluator("
            f"use_cuda={self.use_cuda}, "
            f"debug={self.debug}, "
            ")"
        )
    
    def evaluate(self, epoch):
        self.model.eval()
        r = {
            "_meta": {"type": "validation"},
            "E": epoch,
            "Length": 0,
            "Loss": 0,
        }
        for name in self.metrics:
            r[name] = 0
        
        with torch.no_grad():
            for _, (data, target) in enumerate(self.data_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                r["Loss"] += self.loss_func(output, target).item() * len(target)
                r["Length"] += len(target)
                
                for name, metric in self.metrics.items():
                    r[name] += metric(output, target) * len(target)
        
        for name in self.metrics:
            r[name] /= r["Length"]
        r["Loss"] /= r["Length"]
        
        # Output to file
        self.json_logger.info(r)
        self.debug_logger.info(
            f"\n=> Eval Loss={r['Loss']:.4f} "
            + " ".join(name + "=" + "{:>8.4f}".format(r[name]) for name in self.metrics)
            + "\n"
        )
