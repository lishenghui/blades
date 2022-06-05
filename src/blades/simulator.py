import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from time import time
from typing import Any, Callable, Optional, Union, Iterable
import numpy as np
import ray
import torch
from ray.train import Trainer
import importlib
from .client import TorchClient
from .datasets import FLDataset
from .server import TorchServer
from .utils import top1_accuracy, initialize_logger
from . import alieclient
import sys
sys.path.insert(0,'..')
from aggregators.mean import Mean

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
            result = clients[i].evaluate(
                round_number=round_number,
                test_set=data[i],
                batch_size=batch_size,
                use_actor=use_actor
            )
            update.append(result)
        return update


class Simulator(object):
    """Synchronous and parallel training with specified aggregators."""
    def __init__(
            self,
            dataset: FLDataset,
            aggregator: Callable[[list], torch.Tensor],
            num_byzantine: Optional[int] = 0,
            attack: Optional[str] = 'None',
            num_actors: Optional[int] = 4,
            mode: Optional[str] = 'actor',
            log_interval: Optional[int] = None,
            log_path: str = "./outputs",
            metrics: Optional[dict] = None,
            use_cuda: Optional[bool] = False,
            debug: Optional[bool] = False,
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
        self.device = torch.device("cuda" if use_cuda else "cpu")
        gpu_per_actor = kwargs["gpu_per_actor"] if "gpu_per_actor" in kwargs else 0
        self.log_path = log_path
        initialize_logger(log_path)
        self.use_actor = True if mode == 'actor' else False
        self.aggregator = Mean()
        # self.server_opt = torch.optim.SGD(model.parameters(), lr=lr)
        # self.server = TorchServer(self.server_opt, model=model)
        if type(dataset) != FLDataset:
            traindls, testdls = dataset.get_dls()
            dataset = FLDataset(traindls, testdls)
        self.dataset = dataset
        self.log_interval = log_interval
        self.metrics = {"top1": top1_accuracy} if metrics is None else metrics
        self.use_cuda = use_cuda
        self.debug = debug
        self.omniscient_callbacks = []
        self.random_states = {}
        
        self.json_logger = logging.getLogger("stats")
        self.debug_logger = logging.getLogger("debug")
        self.debug_logger.info(self.__str__())
        
        
        self._setup_clients(attack, num_byzantine=num_byzantine)
        if metrics is None:
            metrics = {"top1": top1_accuracy}
        
        if self.use_actor:
            self.ray_actors = [RayActor.options(num_gpus=gpu_per_actor).remote() for _ in range(num_actors)]
        else:
            self.executor = ThreadPoolExecutor(max_workers=num_trainers)
            self.ray_trainers = [Trainer(backend="torch", num_workers=num_actors // num_trainers, use_gpu=use_cuda,
                                         resources_per_worker={'GPU': gpu_per_actor}) for _ in range(num_trainers)]
            [trainer.start() for trainer in self.ray_trainers]
    
    def _setup_clients(self, attack: str, num_byzantine):
        # from pathlib import Path
        # abs_path = Path(__file__).absolute().parent.parent
        # module_path = importlib.import_module('attackers.%sclient' % attack, abs_path)
        # attack_scheme = getattr(module_path, '%sClient' % attack.capitalize())
        users = self.dataset.get_clients()
        self.clients = []
        for i, u in enumerate(users):
            if i < num_byzantine:
                client = alieclient.AlieClient(20, 5, client_id=u, metrics=self.metrics, device=self.device)
                # client = attack_scheme(20, 5, client_id=u, metrics=self.metrics, device=self.device)
            else:
                client = TorchClient(u, metrics=self.metrics, device=self.device)
            self.clients.append(client)
    
    
    def cache_random_state(self) -> None:
        # This function should be used for reproducibility
        if self.use_cuda:
            self.random_states["torch_cuda"] = torch.cuda.get_rng_state()
        self.random_states["torch"] = torch.get_rng_state()
        self.random_states["numpy"] = np.random.get_state()
    
    def restore_random_state(self) -> None:
        # This function should be used for reproducibility
        if self.use_cuda:
            torch.cuda.set_rng_state(self.random_states["torch_cuda"])
        torch.set_rng_state(self.random_states["torch"])
        np.random.set_state(self.random_states["numpy"])
    
    def register_omniscient_callback(self, callback):
        self.omniscient_callbacks.append(callback)
    
    # clients definition is moved into .run() function, (Tianru)
    # in order to define loss_func, optimizer, model, metrics and etc. in .run()
    '''
    def _setup_clients(self, model, loss_func, device, optimizer, **kwargs):
        users = self.dataset.get_clients()
        self.clients = []
        for i, u in enumerate(users):
            client = TorchClient(u, metrics=self.metrics,
                                 model=model, loss_func=loss_func, device=device, optimizer=optimizer, **kwargs)
            self.clients.append(client)
    '''
    
    def register_attackers(self, client, num):
        # TODO(Shenghui): implement this function to register malicious clients
        pass
    
    def parallel_call(self, clients, f: Callable[[TorchClient], None]) -> None: #clients is added due to the changing of self.clients
        self.cache_random_state()
        _ = [f(worker) for worker in clients]
        self.restore_random_state()
    
    def parallel_get(self, clients, f: Callable[[TorchClient], Any]) -> list: #clients is added due to the changing of self.clients
        results = []
        for w in clients:
            self.cache_random_state()
            results.append(f(w))
            self.restore_random_state()
        return results
    
    def train_actor(self, global_round, num_rounds, clients):
        # TODO: randomly select a subset of clients for local training
        # clients is added due to the changing of self.clients (Tianru)
        # self.clients is changed to clients (Tianru)
        self.debug_logger.info(f"Train global round {global_round}")
        
        def train_function(clients, actor, model, num_rounds):
            data = [self.dataset.get_train_data(client.id, num_rounds) for client in clients]
            return actor.local_training.remote(clients, model, data, num_rounds, use_actor=self.use_actor)
        
        client_per_trainer = len(clients) // len(self.ray_actors)
        all_tasks = [
            train_function(clients[i * client_per_trainer:(i + 1) * client_per_trainer], self.ray_actors[i],
                           self.server.get_model().to('cpu'), num_rounds) for i in range(len(self.ray_actors))]
        
        updates = [item for actor_return in ray.get(all_tasks) for item in actor_return]

        # TODO(Shenghui): This block should be modified to assign update using member function of client.
        for client, update in zip(clients, updates):
            client.state['saved_update'] = update

        # If there are Byzantine workers, ask them to craft attackers based on the updated settings.
        for omniscient_attacker_callback in self.omniscient_callbacks:
            omniscient_attacker_callback()

        updates = self.parallel_get(clients, lambda w: w.get_update())
        
        aggregated = self.aggregator(updates)
        
        self.server.apply_update(aggregated)
        
        self.log_variance(global_round, updates)

    def train_trainer(self, epoch, num_rounds, clients):
        # clients is added due to the changing of self.clients (Tianru)
        # self.clients is changed to clients (Tianru)
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
            data = [self.dataset.get_train_data(client.id, num_rounds) for client in clients]
            return trainer.run(local_training,
                               config={'client': clients, 'data': data, 'model': model, 'use_actor': self.use_actor,
                                       'local_round': num_rounds})
    
        client_per_trainer = len(clients) // len(self.ray_trainers)
        all_tasks = [
            self.executor.submit(train_function, clients[i * client_per_trainer:(i + 1) * client_per_trainer],
                                 self.ray_trainers[i],
                                 self.server.get_model().to('cpu'), num_rounds) for i in range(len(self.ray_trainers))]
    
        update = []
        for task in as_completed(all_tasks):
            update.extend(task.result()[0])
    
        aggregated = self.aggregator(update)
    
        self.server.apply_update(aggregated)
    
        self.log_variance(epoch, update)
         
    def test_actor(self, global_round, batch_size, clients):
        # clients is added due to the changing of self.clients (Tianru)
        # self.clients is changed to clients (Tianru)

        def test_function(clients, actor, model, batch_size):
            data = [self.dataset.get_all_test_data(client.id) for client in clients]
            return actor.evaluate.remote(clients, model, data, round_number=global_round, batch_size=batch_size,
                                         use_actor=self.use_actor)
        
        client_per_trainer = len(clients) // len(self.ray_actors)
        all_tasks = [
            test_function(clients[i * client_per_trainer:(i + 1) * client_per_trainer], self.ray_actors[i],
                          self.server.get_model().to('cpu'), batch_size) for i in range(len(self.ray_actors))]
        
        metrics = [item for actor_return in ray.get(all_tasks) for item in actor_return]
        
        loss, top1 = self.log_validate(metrics)
        # print(f"Test global round {global_round}, loss: {loss}, top1: {top1}")
        self.debug_logger.info(f"Test global round {global_round}, loss: {loss}, top1: {top1}")
    
    def run(
            self,
            model,
            server_optimizer: Union[torch.optim.Optimizer, str] = 'SGD',
            client_optimizer: Union[torch.optim.Optimizer, str] = 'SGD',
            loss: Optional[str] = 'crossentropy',
            global_rounds: Optional[int] = 1,
            local_steps: Optional[int] = 1,
            validate_interval: Optional[int] = 1,
            test_batch_size: Optional[int] = 64,
            lr: Optional[float] = 0.1,
            lr_scheduler = None,
    ):
        
        if server_optimizer == 'SGD':
            self.server_opt = torch.optim.SGD(model.parameters(), lr=lr)
        else:
            raise NotImplementedError
            
        self.client_opt = client_optimizer
        self.server = TorchServer(self.server_opt, model=model)

        self.parallel_call(self.clients, lambda client: client.set_loss(loss))
        global_start = time()
        ret = []
        for global_rounds in range(1, global_rounds + 1):
            self.parallel_call(self.clients, lambda client: client.set_model(self.server.get_model(), torch.optim.SGD, lr))
            round_start = time()
            if self.use_actor:
                self.train_actor(global_rounds, local_steps, self.clients)
            else:
                self.train_trainer(global_rounds, local_steps, self.clients)
                
            if self.use_actor and global_rounds % validate_interval == 0:
                self.test_actor(global_round=global_rounds, batch_size=test_batch_size, clients=self.clients)
                
            # TODO(Shenghui): When using trainer, the test function is not implemented so far.
            if lr_scheduler:
                lr_scheduler.step()
                lr = lr_scheduler.get_last_lr()[0]
            else:
                lr = self.server_opt.param_groups[0]['lr']
            
            ret.append(time() - round_start)
            print(f"E={global_rounds}; Learning rate = {lr:}; Time cost = {time() - global_start}")
        return ret

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
            "Length": np.sum([metric['Length'] for metric in metrics]),
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
