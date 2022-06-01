import torch
import numpy as np
from byzantinefl.models.cifar10.cct import create_model
from byzantinefl.aggregators.mean import Mean
from byzantinefl.simulator.simulator import Simulator
from byzantinefl.builtinDataset.CIFAR10 import CIFAR10
from byzantinefl.simulator.datasets import FLDataset

from args import parse_arguments
options = parse_arguments()


def main():
    random_seed = 0
    device = torch.device("cpu")

    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    model, loss_func = create_model()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)

    traindls, testdls = CIFAR10(data_root="./data", train_bs=32, num_clients=1200).get_dls()
    scale = [16, 32, 64, 128, 256, 512][0]
    datasets = FLDataset(traindls[:scale], testdls[:scale])
    trainer = Simulator(
        aggregator=Mean(),
        model=model,
        dataset=datasets,
        log_interval=10,
        mode='actor',
        num_actors=8,
        log_path=f'./outputs/clients-{scale}',
    )
    results = trainer.run(model=model, loss_func=loss_func, device=device,
                          optimizer=opt, global_round=100, local_round=1)
    with open(f'./outputs/clients-{scale}/result.csv', "w") as fp:
        fp.write("\n".join(list(map(str, results))))


if __name__ == "__main__":
    import ray

    if not ray.is_initialized():
        ray.init(include_dashboard=True, num_gpus=0, num_cpus=8)
    main()