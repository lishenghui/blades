import torch
import numpy as np
from models.MNIST.dnn import create_model
from aggregators.mean import Mean
from simulator.simulator import Simulator
from builtinDataset.MNIST import MNIST
from simulator.datasets import FLDataset
from simulator.utils import top1_accuracy


def main():
    random_seed = 0

    device = torch.device("cpu")

    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    model, loss_func = create_model()
    metrics = {"top1": top1_accuracy}

    traindls, testdls = MNIST(data_root="./data", train_bs=32, num_clients=10).get_dls()
    datasets = FLDataset(traindls, testdls)
    trainer = Simulator(
        aggregator=Mean(),
        model=model,
        loss_func=loss_func,
        dataset=datasets,
        log_interval=10,
        metrics=metrics,
        debug=False,
        gpu_per_actor=0,
        num_trainers=5,
        device=device,
        mode='actor'
    )
    trainer.run(global_round=100, local_round=1)


if __name__ == "__main__":
    import ray

    if not ray.is_initialized():
        # ray.init(local_mode=True, include_dashboard=True, num_gpus=options.num_gpus)
        ray.init(include_dashboard=True, num_gpus=0)
    main()

