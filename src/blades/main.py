import importlib

import numpy as np
import torch
from torch.nn.modules.loss import CrossEntropyLoss

from args import parse_arguments
from datasets.CIFAR10 import CIFAR10
from simulator.datasets import FLDataset
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
    
    model_path = importlib.import_module(options.model_path)
    Model = getattr(model_path, "Net")
    model = Model()
    loss_func = CrossEntropyLoss()
    opt = torch.optim.SGD(model.parameters(), lr=0.2)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            opt, milestones=[100, 200, 400], gamma=0.5
        )
    metrics = {"top1": top1_accuracy}
    
    train_dls, testdls = CIFAR10(data_root=options.data_dir, train_bs=options.batch_size).get_dls()
    dataset = FLDataset(train_dls, testdls)
    trainer = Simulator(
        aggregator=agg_scheme(options),
        model=model,
        dataset=dataset,
        log_interval=args.log_interval,
        metrics=metrics,
        use_cuda=args.use_cuda,
        debug=False,
        num_trainers=args.num_trainers,
        gpu_per_actor=args.gpu_per_actor,
        num_actors=args.num_actors,
        device=device,
        mode=args.mode,
    )
    
    trainer.run(model=model, 
        loss_func=loss_func, 
        validate_interval=20, 
        device=device, 
        optimizer=opt, 
        lr_scheduler=lr_scheduler, 
        global_round=600, 
        local_round=50,
    )


if __name__ == "__main__":
    import ray
    
    if not ray.is_initialized():
        # ray.init(local_mode=True, include_dashboard=True, num_gpus=options.num_gpus)
        ray.init(include_dashboard=True, num_gpus=options.num_gpus)
    main(options)
