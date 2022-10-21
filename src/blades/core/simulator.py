import copy
import importlib
import logging
from time import time
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import ray
import torch
from ray.util import ActorPool
from tqdm import trange

# from blades.utils.torch_utils import parameters_to_vector
from blades.clients import RSAClient
from blades.clients.client import BladesClient, ByzantineClient
from blades.core.actor import _RayActor
from blades.datasets.fldataset import FLDataset
from blades.servers import BladesServer, RSAServer
from blades.utils.utils import (
    initialize_logger,
    # reset_model_weights,
    # set_random_seed,
    top1_accuracy,
)


class Simulator(object):
    """Synchronous and parallel training with specified aggregators.

    :param dataset: FLDataset that consists local data of all input
    :param aggregator: String (name of built-in aggregation scheme) or
                       a callable which takes a list of tensors and returns an
                       aggregated tensor.
    :param num_byzantine: Number of Byzantine input under built-in attack.
                          It should be `0` if you have custom attack strategy.
    :type num_byzantine: int, optional
    :param attack: ``None`` by default. One of the built-in attacks, i.e.,
                    ``None``, ``noise``, ``labelflipping``,
                    ``signflipping``, ``alie``,
                    ``ipm``.
                    It should be ``None`` if you have custom attack strategy.
    :type attack: str
    :param num_actors: Number of ``Ray actors`` that will be created for local
                training.
    :type num_actors: int
    :param mode: Training mode, either ``actor`` or ``trainer``. For large
            scale client population, ``actor mode`` is favorable
    :type mode: str
    :param log_path: The path of logging
    :type log_path: str
    """

    def __init__(
        self,
        dataset: FLDataset,
        global_model: torch.nn.Module,
        *,
        configs=None,
        num_byzantine: Optional[int] = 0,
        attack: Optional[str] = None,
        attack_kws: Optional[Dict[str, float]] = None,
        adversary_kws: Optional[Dict[str, float]] = None,
        aggregator: Union[Callable[[list], torch.Tensor], str] = "mean",
        aggregator_kws: Optional[Dict[str, float]] = None,
        num_actors: Optional[int] = 1,
        gpu_per_actor: Optional[float] = 0,
        log_path: str = "./outputs",
        metrics: Optional[dict] = None,
        use_cuda: Optional[bool] = False,
        seed: Optional[int] = None,
        **kwargs,
    ):

        if configs is None:
            configs = {}
        self.configs = configs

        if adversary_kws is None:
            adversary_kws = {}
        if use_cuda or ("gpu_per_actor" in kwargs and kwargs["gpu_per_actor"] > 0.0):
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        if aggregator_kws is None:
            aggregator_kws = {}
        self._init_aggregator(aggregator=aggregator, aggregator_kws=aggregator_kws)

        # Setup logger
        initialize_logger(log_path)
        self.global_model = global_model
        self.metrics = {"top1": top1_accuracy} if metrics is None else metrics
        self.json_logger = logging.getLogger("stats")
        self.debug_logger = logging.getLogger("debug")
        self.debug_logger.info(self.__str__())

        self.random_states = {}
        self.omniscient_callbacks = []

        if kwargs:
            # User passed in extra keyword arguments but isn't connecting
            # through the simulator. Raise an error, since most likely a typo
            # in keyword
            unknown = ", ".join(kwargs)
            raise RuntimeError(f"Unknown keyword argument(s): {unknown}")

        self.ray_actors = [
            _RayActor.options(num_gpus=gpu_per_actor).remote(dataset)
            for _ in range(num_actors)
        ]
        self.actor_pool = ActorPool(self.ray_actors)

        self.dataset = dataset
        if attack_kws is None:
            attack_kws = {}
        self._setup_clients(
            attack,
            num_byzantine=num_byzantine,
            attack_kws=attack_kws,
        )
        self._setup_adversary(attack, adversary_kws=adversary_kws)

        # set_random_seed(seed)

    def _init_aggregator(self, aggregator, aggregator_kws):
        if type(aggregator) == str:
            agg_path = importlib.import_module("blades.aggregators.%s" % aggregator)
            agg_scheme = getattr(agg_path, aggregator.capitalize())
            self.aggregator = agg_scheme(**aggregator_kws)
        else:
            self.aggregator = aggregator

    def _setup_adversary(self, attack: str, adversary_kws):
        module_path = importlib.import_module("blades.attackers.%sclient" % attack)
        adversary_cls = getattr(
            module_path, "%sAdversary" % attack.capitalize(), lambda: None
        )
        self.adversary = adversary_cls(**adversary_kws) if adversary_cls else None

    def _setup_clients(self, attack: str, num_byzantine, attack_kws):
        import importlib

        if attack is None:
            num_byzantine = 0
        users = self.dataset.get_clients()
        self._clients = {}
        for i, u in enumerate(users):
            u = str(u)
            if i < num_byzantine:
                module_path = importlib.import_module(
                    "blades.attackers.%sclient" % attack
                )
                attack_scheme = getattr(module_path, "%sClient" % attack.capitalize())
                client = attack_scheme(id=u, device=self.device, **attack_kws)
                self._register_omniscient_callback(client.omniscient_callback)
            else:
                if self.configs.client == "RSA":
                    per_model = copy.deepcopy(self.global_model)
                    per_opt = torch.optim.SGD(per_model.parameters(), lr=1.0)
                    client = RSAClient(
                        per_model,
                        per_opt,
                        lambda_=0.1,
                        id=u,
                        device=self.device,
                    )
                else:
                    client = BladesClient(id=u, device=self.device)
            self._clients[u] = client

    def _register_omniscient_callback(self, callback):
        self.omniscient_callbacks.append(callback)

    def get_clients(self) -> List:
        r"""Return all clients as a list."""
        return list(self._clients.values())

    def set_trusted_clients(self, ids: List[str]) -> None:
        r"""Set a list of input as trusted. This is usable for trusted-based
        algorithms that assume some clients are known as not Byzantine.

        :param ids: a list of client ids that are trusted
        :type ids: list
        """
        for id in ids:
            self._clients[id].trust()

    def cache_random_state(self) -> None:
        # This function should be used for reproducibility
        if self.device != torch.device("cpu"):
            self.random_states["torch_cuda"] = torch.cuda.get_rng_state()
        self.random_states["torch"] = torch.get_rng_state()
        self.random_states["numpy"] = np.random.get_state()

    def restore_random_state(self) -> None:
        # This function should be used for reproducibility
        if self.device != torch.device("cpu"):
            torch.cuda.set_rng_state(self.random_states["torch_cuda"])
        torch.set_rng_state(self.random_states["torch"])
        np.random.set_state(self.random_states["numpy"])

    def register_attackers(
        self, clients: List[ByzantineClient], replace_indices=None
    ) -> None:
        r"""Register a list of clients as attackers. Those malicious clients
        replace the first few clients.

        Args:
            clients:  a list of Byzantine clients that will replace some of
                        honest ones.
            replace_indices:  a list of indices of clients to be replaced by
                                the Byzantine clients. The length of this
                                list should be equal to that of ``clients``
                                parameter. If it remains ``None``, the first
                                ``n`` clients will be replaced, where ``n`` is
                                 the length of ``clients``.
        """
        if replace_indices:
            assert len(clients) < len(replace_indices)
        else:
            replace_indices = list(range(len(clients)))
        assert len(clients) < len(self._clients)

        client_li = self.get_clients()
        for i in replace_indices:
            id = client_li[i].id()
            clients[i].set_id(id)
            self._clients[id] = clients[i]
            self._register_omniscient_callback(clients[i].omniscient_callback)

    def parallel_call(self, clients, f: Callable[[BladesClient], None]) -> None:
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

    def train_actor(
        self,
        global_round: int,
        num_rounds: int,
        clients: List[BladesClient],
        lr: float,
        *args,
        **kwargs,
    ) -> None:
        r"""Run local training using ``ray`` actors.

        Args:
            global_round (int): The current global round.
            num_rounds (int): The number of local update steps.
            clients (list): A list of clients that perform local training.
            lr (float): Learning rate for client optimizer.
        """

        # TODO: randomly select a subset of input for local training
        self.debug_logger.info(f"Train global round {global_round}")

        # Allocate input to actors:
        global_model = ray.put(self.server.get_model())
        client_groups = np.array_split(self.get_clients(), len(self.ray_actors))
        all_results = []
        for clients, actor in zip(client_groups, self.ray_actors):
            ref_clients = actor.local_training.remote(
                clients=clients,
                global_model=global_model,
                local_round=num_rounds,
                lr=lr,
                *args,
                **kwargs,
            )
            all_results.append(ref_clients)

        clients = [
            client for client_group in all_results for client in ray.get(client_group)
        ]

        for client in clients:
            self._clients[client._id] = client

        if self.adversary:
            self.adversary.omniscient_callback(self)

        # If there are Byzantine workers, ask them to craft attackers based on
        # the updated settings.
        for omniscient_attacker_callback in self.omniscient_callbacks:
            omniscient_attacker_callback(self)

        self.server.global_update(clients)
        # self.log_variance(global_round, updates)

    def test_actor(self, global_round, batch_size):
        """Evaluates the global global_model using test set.

        Args:
            global_round: the current global round number
            batch_size: test batch size

        Returns:
        """
        global_model = self.server.get_model()
        client_groups = np.array_split(self.get_clients(), len(self.ray_actors))

        all_results = self.actor_pool.map(
            lambda actor, clients: actor.evaluate.remote(
                clients=clients,
                global_model=global_model,
                round_number=global_round,
                batch_size=batch_size,
                metrics=self.metrics,
            ),
            client_groups,
        )

        metrics = [update for returns in list(all_results) for update in returns]

        loss, top1 = self.log_validate(metrics)
        self.debug_logger.info(
            f"Test global round {global_round}, loss: {loss}, top1: {top1}"
        )
        return loss, top1

    def log_variance(self, cur_round, update):
        updates = []
        for client in self._clients.values():
            if not client.is_byzantine():
                updates.append(client.get_update())
        mean_update = torch.mean(torch.vstack(updates), dim=0)
        var_avg = torch.mean(
            torch.var(torch.vstack(updates), dim=0, unbiased=False)
        ).item()
        norm = torch.norm(
            torch.var(torch.vstack(updates), dim=0, unbiased=False)
        ).item()
        avg_norm = torch.norm(mean_update)
        var_norm = torch.sqrt(
            torch.mean(
                torch.tensor(
                    [
                        torch.norm(model_update - mean_update) ** 2
                        for model_update in updates
                    ]
                )
            )
        )

        r = {
            "_meta": {"type": "variance"},
            "Round": cur_round,
            "avg": var_avg,
            "norm": norm,
            "avg_norm": avg_norm,
            "VN_ratio": var_norm / avg_norm,
        }

        # Output to file
        self.json_logger.info(r)

    def log_validate(self, metrics):
        top1 = np.average(
            [metric["top1"] for metric in metrics],
            weights=[metric["Length"] for metric in metrics],
        )
        loss = np.average(
            [metric["Loss"] for metric in metrics],
            weights=[metric["Length"] for metric in metrics],
        )
        r = {
            "_meta": {"type": "test"},
            "Round": metrics[0]["E"],
            "top1": top1,
            "Length": np.sum([metric["Length"] for metric in metrics]),
            "Loss": loss,
        }
        self.json_logger.info(r)
        return loss, top1

    def log_train(self, progress, batch_idx, epoch, results):
        length = sum(res["length"] for res in results)

        r = {
            "_meta": {"type": "train"},
            "Round": epoch,
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
            f"[Round{r['E']:2}B{r['B']:<3}| {progress:6}/{total} ({pct:3.0f}%)"
            f" ] Loss: {r['Loss']:.4f} "
            + " ".join(name + "=" + "{:>8.4f}".format(r[name]) for name in self.metrics)
        )
        # Output to file
        self.json_logger.info(r)

    def run(
        self,
        server_optimizer: Union[torch.optim.Optimizer, str] = "SGD",
        client_optimizer: Union[torch.optim.Optimizer, str] = "SGD",
        loss: Optional[str] = "crossentropy",
        global_rounds: Optional[int] = 1,
        local_steps: Optional[int] = 1,
        validate_interval: Optional[int] = 1,
        test_batch_size: Optional[int] = 64,
        server_lr: Optional[float] = 0.1,
        client_lr: Optional[float] = 0.1,
        server_lr_scheduler: Optional[torch.optim.lr_scheduler.MultiStepLR] = None,
        client_lr_scheduler: Optional[torch.optim.lr_scheduler.MultiStepLR] = None,
        dp_kws: Optional[Dict[str, float]] = None,
    ):
        """Run the adversarial training.

        :param global_model: The global global_model for training.
        :type global_model: torch.nn.Module
        :param server_optimizer: The optimizer for server-side optimization.
                                urrently, the `str` type only supports ``SGD``
        :type server_optimizer: torch.optim.Optimizer or str
        :param client_optimizer: Optimizer for client-side optimization.
                            Currently, the ``str`` type only supports ``SGD``
        :type client_optimizer: torch.optim.Optimizer or str
        :param loss: A Pytorch Loss function. See `Python documentation
        <https://pytorch.org/docs/stable/nn.html#loss-functions>`_.
                     Currently, the `str` type only supports ``crossentropy``
        :type loss: str
        :param global_rounds: Number of communication rounds in total.
        :type global_rounds: int, optional
        :param local_steps: Number of local steps of each global round.
        :type local_steps: int, optional
        :param validate_interval: Interval of evaluating the global global_model using
        test set.
        :type validate_interval: int, optional
        :param test_batch_size: Batch size of evaluation
        :type test_batch_size: int, optional
        :param server_lr: Learning rate of ``server_optimizer``
        :type server_lr: float, optional
        :param client_lr: Learning rate of ``client_optimizer``
        :type client_lr: float, optional
        :param server_lr_scheduler: Server learning rate scheduler
        :type server_lr_scheduler: torch.optim.lr_scheduler.MultiStepLR
        :param client_lr_scheduler: Client learning rate scheduler
        :type client_lr_scheduler: torch.optim.lr_scheduler.MultiStepLR
        :return: None
        """
        if dp_kws:
            dp_kws.update({"dp": True})
        else:
            dp_kws = {}

        global_model = self.global_model
        if self.device != torch.device("cpu"):
            global_model = global_model.to("cuda")

        # reset_model_weights(global_model)
        if server_optimizer == "SGD":
            self.server_opt = torch.optim.SGD(
                global_model.parameters(), lr=server_lr, **dp_kws
            )
        else:
            self.server_opt = server_optimizer

        self.client_opt = client_optimizer

        if self.configs.server == "RSA":
            self.server = RSAServer(
                optimizer=self.server_opt,
                model=global_model,
                aggregator=self.aggregator,
            )
        else:
            self.server = BladesServer(
                optimizer=self.server_opt,
                model=global_model,
                aggregator=self.aggregator,
            )
        self.parallel_call(self.get_clients(), lambda client: client.set_loss(loss))
        global_start = time()
        ret = []
        global_model = self.server.get_model()

        for actor in self.ray_actors:
            actor.set_global_model.remote(global_model, torch.optim.SGD, client_lr)

        with trange(0, global_rounds + 1) as t:
            for global_rounds in t:
                round_start = time()

                self.train_actor(
                    global_rounds, local_steps, self.get_clients(), client_lr, **dp_kws
                )
                if server_lr_scheduler:
                    server_lr_scheduler.step()

                if client_lr_scheduler:
                    client_lr_scheduler.step()
                    client_lr = client_lr_scheduler.get_last_lr()[0]

                ret.append(time() - round_start)
                server_lr = self.server.get_opt().param_groups[0]["lr"]
                if global_rounds % validate_interval == 0:
                    loss, top1 = self.test_actor(
                        global_round=global_rounds, batch_size=test_batch_size
                    )
                    t.set_postfix(loss=loss, top1=top1)
                self.debug_logger.info(
                    f"E={global_rounds}; Server learning rate = {server_lr:}; "
                    f"Client learning rate = {client_lr:}; Time cost = "
                    f"{time() - global_start}"
                )

            return ret
