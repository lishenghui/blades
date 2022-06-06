import numpy as np
import torch
from aggregators.mean import Mean
from args import parse_arguments
from datasets.MNIST import MNIST
from models.MNIST.dnn import create_model
from simulator.datasets import FLDataset
from simulator.simulator import Simulator

options = parse_arguments()


def main():
    random_seed = 0
    device = torch.device("cpu")
    
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    
    model, loss_func = create_model()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    
    traindls, testdls = MNIST(data_root="./data", train_bs=32, num_clients=10).get_dls()
    datasets = FLDataset(traindls, testdls)
    trainer = Simulator(
        aggregator=Mean(),
        model=model,
        loss_func=loss_func,
        dataset=datasets,
        log_interval=10,
        gpu_per_actor=0,
        device=device,
        mode='actor',
        log_path='./outputs/mnist',
    )
    trainer.run(model=model, loss=loss_func, device=device, optimizer=opt, global_rounds=100, local_steps=20)


if __name__ == "__main__":
    import ray
    
    if not ray.is_initialized():
        ray.init(include_dashboard=True, num_gpus=0)
    main()
