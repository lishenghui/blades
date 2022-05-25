import importlib

import numpy as np
import torch
from torch.nn.modules.loss import CrossEntropyLoss

from args import parse_arguments
from simulator.datasets import CIFAR10
# current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
# parent_dir = os.path.dirname(current_dir)
# sys.path.insert(0, parent_dir)
from simulator.server import TorchServer
from simulator.simulator import Simulator
from simulator.utils import top1_accuracy, initialize_logger

options = parse_arguments()

agg_path = importlib.import_module('aggregators.%s' % options.agg)
agg_scheme = getattr(agg_path, options.agg.capitalize())


def main(args):
    initialize_logger(options.log_dir)
    device = torch.device("cuda" if args.use_cuda else "cpu")
    
    kwargs = {"pin_memory": True} if args.use_cuda else {}
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    opt = importlib.import_module(options.model_path)
    Model = getattr(opt, "Net")
    model = Model().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=options.lr)
    loss_func = CrossEntropyLoss().to(device)
    
    metrics = {"top1": top1_accuracy}
    
    server_opt = torch.optim.SGD(model.parameters(), lr=options.lr)
    server = TorchServer(server_opt, model=model)
    dataset = CIFAR10(data_root=options.data_dir, train_bs=options.batch_size)
    trainer = Simulator(
        server=server,
        aggregator=agg_scheme(options),
        model=model,
        dataset=dataset,
        max_batches_per_epoch=options.local_round,
        log_interval=args.log_interval,
        metrics=metrics,
        use_cuda=args.use_cuda,
        debug=False,
        num_trainers=args.num_trainers,
        gpu_per_actor=args.gpu_per_actor,
        num_actors=args.num_actors,
        use_actor=args.use_actor
    )
    
    trainer.setup_clients(model, loss_func, device, optimizer)
    trainer.train(round=100, local_round=options.local_round)


if __name__ == "__main__":
    import ray
    
    if not ray.is_initialized():
        # ray.init(local_mode=True, include_dashboard=True, num_gpus=options.num_gpus)
        ray.init(include_dashboard=True, num_gpus=options.num_gpus)
    main(options)
