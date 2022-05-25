import logging
import os
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Union, Callable, Any

import numpy as np
import ray
import torch
from ray.train import Trainer

from .client import TorchClient
from .server import TorchServer


@ray.remote
class RayActor(object):
    def __int__(*args, **kwargs):
        super().__init__()
    
    def local_training(self, clients, model, data, local_round, use_actor=False):
        update = []
        for i in range(len(clients)):
            clients[i].set_para(model)
            clients[i].train_epoch_start()
            clients[i].local_training(local_round, use_actor, data[i])
            update.append(clients[i].get_update())
        return update
    
    def evaluate(self, clients, model, data, round_number, batch_size, use_actor=False):
        update = []
        for i in range(len(clients)):
            clients[i].set_para(model)
            result = clients[i].evaluate(round_number=round_number, test_set=data[i], batch_size=batch_size,
                                         use_actor=use_actor)
            update.append(result)
        return update


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
    
    def setup_clients(self, data_path, model, loss_func, device, optimizer, **kwargs):
        assert os.path.isfile(data_path)
        with open(data_path, 'rb') as f:
            (users, train_data, test_clients, test_data) = [pickle.load(f) for _ in range(4)]
        print(users)
        self.clients = []
        for i, u in enumerate(users):
            client = TorchClient(u, metrics=self.metrics,
                                 model=model, loss_func=loss_func, device=device, optimizer=optimizer, **kwargs)
            self.clients.append(client)
    
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



class ParallelTrainer(DistributedSimulatorBase):
    """Synchronous and parallel training with specified aggregators."""
    
    def __init__(
            self,
            server: TorchServer,
            data_manager,
            aggregator: Callable[[list], torch.Tensor],
            log_interval: int,
            metrics: dict,
            use_cuda: bool,
            debug: bool,
            pre_batch_hooks=None,
            post_batch_hooks=None,
            **kwargs
    ):
        """
        Args:
            aggregator (callable): A callable which takes a list of tensors and returns
                an aggregated tensor.
            log_interval (int): Control the frequency of logging training batches
            metrics (dict): dict of metric names and their functions
            use_cuda (bool): Use cuda or not
            debug (bool):
        """
        num_trainers = kwargs["num_trainers"] if "num_trainers" in kwargs else 1
        num_actors = kwargs["num_actors"] if "num_actors" in kwargs else 1
        gpu_per_actor = kwargs["gpu_per_actor"] if "gpu_per_actor" in kwargs else 0
        self.use_actor = kwargs["use_actor"] if "use_actor" in kwargs else 0
        self.aggregator = aggregator
        self.server = server
        self.data_manager = data_manager
        self.pre_batch_hooks = pre_batch_hooks or []
        self.post_batch_hooks = post_batch_hooks or []
        self.log_interval = log_interval
        super().__init__(metrics, use_cuda, debug)
        
        if self.use_actor:
            self.ray_actors = [RayActor.options(num_gpus=gpu_per_actor).remote() for _ in range(num_actors)]
        else:
            self.executor = ThreadPoolExecutor(max_workers=num_trainers)
            self.ray_trainers = [Trainer(backend="torch", num_workers=num_actors // num_trainers, use_gpu=use_cuda,
                                         resources_per_worker={'GPU': gpu_per_actor}) for _ in range(num_trainers)]
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
        # If there are Byzantine workers, ask them to craft attackers based on the updated settings.
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
        # If there are Byzantine workers, ask them to craft attackers based on the updated settings.
        for omniscient_attacker_callback in self.omniscient_callbacks:
            omniscient_attacker_callback()
        
        update = self.parallel_get(lambda w: w.get_update())
        
        aggregated = self.aggregator(update)
        
        self.server.apply_update(aggregated)
    
    # def train(self, epoch):
    #     self.debug_logger.info(f"Train epoch {epoch}")
    #     self.parallel_call(lambda worker: worker.train_epoch_start.remote())
    #
    #     for batch_idx in range(self.max_batches_per_epoch):
    #         try:
    #             self.parallel_call(lambda worker: worker.set_para.remote(self.server.get_model()))
    #             self._run_pre_batch_hooks(epoch, batch_idx)
    #             results = self.parallel_call(lambda w: w.compute_gradient.remote())
    #             # self.aggregation_and_update()
    #             # If there are Byzantine workers, ask them to craft attackers based on the updated settings.
    #             for omniscient_attacker_callback in self.omniscient_callbacks:
    #                 omniscient_attacker_callback()
    #
    #             gradients = self.parallel_get(lambda w: w.get_gradient.remote())
    #             aggregated = self.aggregator(gradients)
    #
    #             # Assume that the model and optimizers are shared among workers.
    #             self.server.set_gradient(aggregated)
    #             self.server.apply_gradient()
    #
    #             # progress += results[0]["length"]
    #             if batch_idx % self.log_interval == 0:
    #                 # self.log_train(progress, batch_idx, epoch, results)
    #                 self.log_variance(epoch, gradients)
    #             self._run_post_batch_hooks(epoch, batch_idx)
    #         except StopIteration:
    #             continue
    
    def train_fedavg_actor(self, global_round, num_rounds):
    
        # TODO: randomly select a subset of clients for local training
        self.debug_logger.info(f"Train global round {global_round}")
        
        def train_function(clients, actor, model, num_rounds):
            data = [self.data_manager.get_train_data(client.id, num_rounds) for client in clients]
            return actor.local_training.remote(clients, model, data, num_rounds, use_actor=self.use_actor)
        
        client_per_trainer = len(self.clients) // len(self.ray_actors)
        all_tasks = [
            train_function(self.clients[i * client_per_trainer:(i + 1) * client_per_trainer], self.ray_actors[i],
                           self.server.get_model().to('cpu'), num_rounds) for i in range(len(self.ray_actors))]
        
        update = [item for actor_return in ray.get(all_tasks) for item in actor_return]
        
        aggregated = self.aggregator(update)
        
        self.server.apply_update(aggregated)
        
        self.log_variance(global_round, update)
    
    def test_actor(self, global_round, batch_size):
        def test_function(clients, actor, model, batch_size):
            data = [self.data_manager.get_all_test_data(client.id) for client in clients]
            return actor.evaluate.remote(clients, model, data, round_number=global_round, batch_size=batch_size,
                                         use_actor=self.use_actor)
        
        client_per_trainer = len(self.clients) // len(self.ray_actors)
        all_tasks = [
            test_function(self.clients[i * client_per_trainer:(i + 1) * client_per_trainer], self.ray_actors[i],
                          self.server.get_model().to('cpu'), batch_size) for i in range(len(self.ray_actors))]
        
        metrics = [item for actor_return in ray.get(all_tasks) for item in actor_return]
        
        loss, top1 = self.log_validate(metrics)
        
        self.debug_logger.info(f"Test global round {global_round}, loss: {loss}, top1: {top1}")


    def train_fedavg(self, epoch, num_rounds):
        
        self.debug_logger.info(f"Train epoch {epoch}")
        
        def local_training(config):
            update = []
            for i in range(len(config['client'])):
                config['client'][i].set_para(config['model'])
                config['client'][i].train_epoch_start()
                config['client'][i].local_training(config['local_round'], config['use_actor'], config['data'][i])
                update.append(config['client'][i].get_update())
            return update
        
        def train_function(clients, trainer, model, num_rounds):
            data = [self.data_manager.get_train_data(client.id, num_rounds) for client in clients]
            return trainer.run(local_training,
                               config={'client': clients, 'data': data, 'model': model, 'use_actor': self.use_actor,
                                       'local_round': num_rounds})
        
        client_per_trainer = len(self.clients) // len(self.ray_trainers)
        all_tasks = [
            self.executor.submit(train_function, self.clients[i * client_per_trainer:(i + 1) * client_per_trainer],
                                 self.ray_trainers[i],
                                 self.server.get_model().to('cpu'), num_rounds) for i in range(len(self.ray_trainers))]
        
        update = []
        for task in as_completed(all_tasks):
            update.extend(task.result()[0])
        
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
    
    def log_validate(self, metrics):
        top1 = np.average([metric['top1'] for metric in metrics], weights=[metric['Length'] for metric in metrics])
        loss = np.average([metric['Loss'] for metric in metrics], weights=[metric['Length'] for metric in metrics])
        r = {
            "_meta": {"type": "test"},
            "E": metrics[0]['E'],
            "top1": top1,
            "Length":np.sum([metric['Length'] for metric in metrics]),
            "Loss": loss,
        }
        self.json_logger.info(r)
        return loss, top1

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
                output = self.model.to(self.device)(data)
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
