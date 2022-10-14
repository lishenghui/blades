import ray
from blades.core.actor import _RayActor
from ray.util import ActorPool
import torch
from torch.multiprocessing.reductions import reduce_tensor
from torch.nn import Module
from blades.datasets.fldataset import FLDataset
from blades.utils.torch_utils import get_num_params


@ray.remote
class ActorManager:
    def __init__(
        self,
        global_model: Module,
        opt: type(torch.optim.Optimizer),
        lr: float,
        num_actors: int,
        gpu_per_actor: float,
        dataset: FLDataset,
    ):
        num_params = get_num_params(global_model)
        self.shared_memory = torch.randn((num_actors, num_params))
        mem_meta_info = reduce_tensor(self.shared_memory)
        self.ray_actors = [
            _RayActor.options(num_gpus=gpu_per_actor).remote(dataset, i, mem_meta_info)
            for i in range(num_actors)
        ]
        self.actor_pool = ActorPool(self.ray_actors)
        # rets = []
        # for actor in self.ray_actors:
        #     model = copy.deepcopy(global_model).share_memory()
        #     opt = copy.deepcopy(opt).shared_memory()
        #     ret = actor.set_global_model.remote(model, opt, lr)
        #     rets.append(ret)
        # ray.get(rets)

    def update(self):
        ray.get(self.ray_actors[0].update.remote())
        print(self.shared_memory)

    def get_shared_updates(self):
        return self.shared_memory
