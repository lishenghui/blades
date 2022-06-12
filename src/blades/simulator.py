import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from time import time
from typing import Any, Callable, Optional, Union, List
import importlib
import numpy as np
import ray
import torch
from ray.train import Trainer
from ray.util import ActorPool

from blades.client import BladesClient, ByzantineClient
from blades.datasets.datasets import FLDataset
from blades.server import BladesServer
from blades.utils import top1_accuracy, initialize_logger


@ray.remote
class _RayActor(object):
    """Ray Actor"""
    
    def __init__(self, dataset: object, *args, **kwargs):
        """
       Args:
           aggregator (callable): A callable which takes a list of tensors and returns
               an aggregated tensor.
           log_interval (int): Control the frequency of logging training batches
           metrics (dict): dict of metric names and their functions
           use_cuda (bool): Use cuda or not
           debug (bool):
       """
        traindls, testdls = dataset.get_dls()
        self.dataset = FLDataset(traindls, testdls)
    
    def local_training(self, clients, model, local_round):
        update = []
        for i in range(len(clients)):
            clients[i].set_para(model)
            clients[i].train_epoch_start()
            data = self.dataset.get_train_data(clients[i].id(), local_round)
            clients[i].local_training(local_round, use_actor=True, data_batches=data)
            update.append(clients[i].get_update())
        return update
    
    def evaluate(self, clients, model, round_number, batch_size, metrics):
        update = []
        for i in range(len(clients)):
            clients[i].set_para(model)
            data = self.dataset.get_all_test_data(clients[i].id())
            result = clients[i].evaluate(
                round_number=round_number,
                test_set=data,
                batch_size=batch_size,
                metrics=metrics,
                use_actor=True,
            )
            update.append(result)
        return update


class Simulator(object):
    """Synchronous and parallel training with specified aggregators.
    
    :param dataset: FLDataset that consists local data of all input
    :param aggregator: String (name of build-in aggregation scheme) or
                       a callable which takes a list of tensors and returns an aggregated tensor.
    :param num_byzantine: Number of Byzantine input under build-in attack.
                          It should be ``0`` if you have custom attack strategy.
    :type num_byzantine: int, optional
    :param attack: ``None`` by default. One of the build-in attacks, i.e., ``None``, ``noise``, ``labelflipping``,
                    ``signflipping``, ``alie``,
                    ``ipm``. It should be ``None`` if you have custom attack strategy.
    :type attack: str
    :param num_actors: Number of ``Ray actors`` that will be created for local training.
    :type num_actors: int
    :param mode: Training mode, either ``actor`` or ``trainer``. For large scale client population,
                    ``actor mode`` is favorable
    :type mode: str
    :param log_path: The path of logging
    :type log_path: str
    
    """
    
    def __init__(
            self,
            dataset: FLDataset,
            aggregator: Union[Callable[[list], torch.Tensor], str],
            num_byzantine: Optional[int] = 0,
            attack: Optional[str] = 'None',
            num_actors: Optional[int] = 4,
            mode: Optional[str] = 'actor',
            log_path: str = "./outputs",
            metrics: Optional[dict] = None,
            use_cuda: Optional[bool] = False,
            **kwargs,
    ):
        
        num_trainers = kwargs["num_trainers"] if "num_trainers" in kwargs else 1
        gpu_per_actor = kwargs["gpu_per_actor"] if "gpu_per_actor" in kwargs else 0
        self.use_actor = True if mode == 'actor' else False
        self.device = torch.device("cuda" if use_cuda or "gpu_per_actor" in kwargs else "cpu")
        attack_param = kwargs["attack_param"] if "attack_param" in kwargs else {}
        initialize_logger(log_path)
        agg_param = kwargs["agg_param"] if "agg_param" in kwargs else {}
        self._init_aggregator(aggregator=aggregator, agg_param=agg_param)
        
        self.metrics = {"top1": top1_accuracy} if metrics is None else metrics
        self.omniscient_callbacks = []
        self.random_states = {}
        
        self.json_logger = logging.getLogger("stats")
        self.debug_logger = logging.getLogger("debug")
        self.debug_logger.info(self.__str__())
        
        if self.use_actor:
            self.ray_actors = [_RayActor.options(num_gpus=gpu_per_actor).remote(dataset)
                               for _ in range(num_actors)]
            self.actor_pool = ActorPool(self.ray_actors)
        else:
            self.executor = ThreadPoolExecutor(max_workers=num_trainers)
            self.ray_trainers = [Trainer(backend="torch", num_workers=num_actors // num_trainers, use_gpu=use_cuda,
                                         resources_per_worker={'GPU': gpu_per_actor}) for _ in range(num_trainers)]
            [trainer.start() for trainer in self.ray_trainers]
        
        if type(dataset) != FLDataset:
            traindls, testdls = dataset.get_dls()
            self.dataset = FLDataset(traindls, testdls)
        
        self._setup_clients(attack, num_byzantine=num_byzantine, attack_param=attack_param)
    
    def _init_aggregator(self, aggregator, agg_param):
        if type(aggregator) == str:
            agg_path = importlib.import_module('blades.aggregators.%s' % aggregator)
            agg_scheme = getattr(agg_path, aggregator.capitalize())
            self.aggregator = agg_scheme(**agg_param)
        else:
            self.aggregator = aggregator
        
    def _setup_clients(self, attack: str, num_byzantine, attack_param):
        import importlib
        if attack is None:
            num_byzantine = 0
        users = self.dataset.get_clients()
        self._clients = {}
        for i, u in enumerate(users):
            # u = str(u)
            if i < num_byzantine:
                module_path = importlib.import_module('blades.attackers.%sclient' % attack)
                attack_scheme = getattr(module_path, '%sClient' % attack.capitalize())
                client = attack_scheme(id=u, device=self.device, **attack_param)
                self._register_omniscient_callback(client.omniscient_callback)
            else:
                client = BladesClient(id=u, device=self.device)
            self._clients[u] = client
    
    def get_clients(self):
        r"""Return all input.
        """
        return self._clients
    
    def set_trusted_clients(self, ids: List[str]) -> None:
        """Set a list of input as trusted. This is usable for trusted-based algorithms that assume some input are known as not Byzantine.
    
        :param ids: a list of client ids that are trusted
        :type ids: list
        """
        for id in ids:
            self._clients[id].trust()
            
    
    def cache_random_state(self) -> None:
        # This function should be used for reproducibility
        if self.device != torch.device('cpu'):
            self.random_states["torch_cuda"] = torch.cuda.get_rng_state()
        self.random_states["torch"] = torch.get_rng_state()
        self.random_states["numpy"] = np.random.get_state()
    
    def restore_random_state(self) -> None:
        # This function should be used for reproducibility
        if self.device != torch.device('cpu'):
            torch.cuda.set_rng_state(self.random_states["torch_cuda"])
        torch.set_rng_state(self.random_states["torch"])
        np.random.set_state(self.random_states["numpy"])
    
    def _register_omniscient_callback(self, callback):
        self.omniscient_callbacks.append(callback)
    
    def register_attackers(self, clients: List[ByzantineClient]) -> None:
        assert len(clients) < len(self._clients)
        client_li = list(self._clients.values())
        for i in range(len(clients)):
            id = client_li[i]._id()
            clients[i].set_id(id)
            self._clients[id] = clients[i]
            self._register_omniscient_callback(clients[i].omniscient_callback)
    
    def parallel_call(self, clients,
                      f: Callable[[BladesClient], None]) -> None:
        self.cache_random_state()
        _ = [f(worker) for worker in clients]
        self.restore_random_state()
    
    def parallel_get(self, clients, f: Callable[[BladesClient], Any]) -> list:
        results = []
        for w in clients:
            self.cache_random_state()
            results.append(f(w))
            self.restore_random_state()
        return results
    
    def train_actor(self, global_round, num_rounds, clients):
        # TODO: randomly select a subset of input for local training
        self.debug_logger.info(f"Train global round {global_round}")
        
        # Allocate input to actors:
        global_model = self.server.get_model().to('cpu')
        client_groups = np.array_split(list(self._clients.values()), len(self.ray_actors))
        all_results = self.actor_pool.map(
            lambda actor, clients:
                actor.local_training.remote(
                    clients=clients,
                    model=global_model,
                    local_round=num_rounds,
                ),
            client_groups
        )
        
        updates = [update for returns in list(all_results) for update in returns]
        for client, update in zip(clients.values(), updates):
            client.save_update(update)
        
        # If there are Byzantine workers, ask them to craft attackers based on the updated settings.
        for omniscient_attacker_callback in self.omniscient_callbacks:
            omniscient_attacker_callback(self)
        
        updates = self.parallel_get(clients.values(), lambda w: w.get_update())
        # aggregated = self.server.aggregator(updates)
        aggregated = self.server.aggregator(clients.values())
        self.server.apply_update(aggregated)
        
        self.log_variance(global_round, updates)
    
    def train_trainer(self, epoch, num_rounds, clients):
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
            data = [self.dataset.get_train_data(client._id, num_rounds) for client in clients]
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
        global_model = self.server.get_model().to('cpu')
        client_groups = np.array_split(list(self._clients.values()), len(self.ray_actors))
        
        all_results = self.actor_pool.map(
            lambda actor, clients:
            actor.evaluate.remote(
                clients=clients,
                model=global_model,
                round_number=global_round,
                batch_size=batch_size,
                metrics=self.metrics,
            ),
            client_groups
        )
        
        metrics = [update for returns in list(all_results) for update in returns]
        
        loss, top1 = self.log_validate(metrics)
        self.debug_logger.info(f"Test global round {global_round}, loss: {loss}, top1: {top1}")
    
    def run(
            self,
            model: torch.nn.Module,
            server_optimizer: Union[torch.optim.Optimizer, str] = 'SGD',
            client_optimizer: Union[torch.optim.Optimizer, str] = 'SGD',
            loss: Optional[str] = 'crossentropy',
            global_rounds: Optional[int] = 1,
            local_steps: Optional[int] = 1,
            validate_interval: Optional[int] = 1,
            test_batch_size: Optional[int] = 64,
            server_lr: Optional[float] = 0.1,
            client_lr: Optional[float] = 0.1,
            lr_scheduler: Optional[torch.optim.lr_scheduler.MultiStepLR] = None,
    ):
        """Run the adversarial training.
    
        :param model: The global model for training.
        :type model: `torch.nn.Module`
        :param server_optimizer: Pytorch optimizer for server-side optimization.
                                    Currently, the `str` type only supports ``SGD``
        :type server_optimizer: torch.optim.Optimizer or str
        :param client_optimizer: Pytorch optimizer for client-side optimization.
                                 Currently, the ``str`` type only supports ``SGD``
        :type client_optimizer: torch.optim.Optimizer or str
        :param loss: A Pytorch Loss function. See `Python documentation <https://pytorch.org/docs/stable/nn.html#loss-functions>`_.
                     Currently, the `str` type only supports ``crossentropy``
        :type loss: str
        :param global_rounds: Number of communication rounds in total.
        :type global_rounds: int, optional
        :param local_steps: Number of local steps of each global round.
        :type local_steps: int, optional
        :param validate_interval: Interval of evaluating the global model using test set.
        :type validate_interval: int, optional
        :param test_batch_size: Batch size of evaluation
        :type test_batch_size: int, optional
        :param server_lr: Learning rate of ``server_optimizer``
        :type server_lr: float, optional
        :param client_lr: Learning rate of ``client_optimizer``
        :type client_lr: float, optional
        :param lr_scheduler: Learning rate scheduler
        :type lr_scheduler: torch.optim.lr_scheduler.MultiStepLR, optional
        :return: None
        """
        if server_optimizer == 'SGD':
            self.server_opt = torch.optim.SGD(model.parameters(), lr=server_lr)
        else:
            self.server_opt = server_optimizer
        
        self.client_opt = client_optimizer
        self.server = BladesServer(optimizer=self.server_opt,
                                   model=model,
                                   aggregator=self.aggregator,
                                   )
        
        self.parallel_call(self._clients.values(), lambda client: client.set_loss(loss))
        global_start = time()
        ret = []
        global_model = self.server.get_model()
        for global_rounds in range(1, global_rounds + 1):
            self.parallel_call(self._clients.values(),
                               lambda client: client.set_model(global_model, torch.optim.SGD, client_lr))
            round_start = time()
            if self.use_actor:
                self.train_actor(global_rounds, local_steps, self._clients)
            else:
                self.train_trainer(global_rounds, local_steps, self._clients)
            
            if self.use_actor and global_rounds % validate_interval == 0:
                self.test_actor(global_round=global_rounds, batch_size=test_batch_size, clients=self._clients)
            
            # TODO(Shenghui): When using trainer, the test method is not implemented so far.
            if lr_scheduler:
                lr_scheduler.step()
                client_lr = lr_scheduler.get_last_lr()[0]
            else:
                client_lr = self.server_opt.param_groups[0]['lr']
            
            ret.append(time() - round_start)
            print(f"E={global_rounds}; Learning rate = {client_lr:}; Time cost = {time() - global_start}")
        return ret
    
    def log_variance(self, cur_round, update):
        var_avg = torch.mean(torch.var(torch.vstack(update), dim=0, unbiased=False)).item()
        norm = torch.norm(torch.var(torch.vstack(update), dim=0, unbiased=False)).item()
        avg_norm = torch.mean(torch.var(torch.vstack(update), dim=0, unbiased=False) / (
            torch.mean(torch.vstack(update) ** 2, dim=0))).item()
        r = {
            "_meta": {"type": "variance"},
            "E": cur_round,
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
        total = len(self._clients.values()[0].data_loader.dataset)
        pct = 100 * progress / total
        self.debug_logger.info(
            f"[E{r['E']:2}B{r['B']:<3}| {progress:6}/{total} ({pct:3.0f}%) ] Loss: {r['Loss']:.4f} "
            + " ".join(name + "=" + "{:>8.4f}".format(r[name]) for name in self.metrics)
        )
        # Output to file
        self.json_logger.info(r)
