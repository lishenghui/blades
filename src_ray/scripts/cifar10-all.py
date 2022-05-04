import inspect
import numpy as np
import os
import sys
import torch
from torchvision import datasets
import ray

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from torch.nn.modules.loss import CrossEntropyLoss
from codes.args import parse_arguments
from codes.sampler import DistributedSampler
from codes.simulators.simulator import (
    ParallelTrainer,
    DistributedEvaluator,
)
from codes.simulators.worker import TorchWorker, WorkerWithMomentum
from codes.simulators.server import TorchServer
from codes.tasks.cifar10 import cifar10
from codes.utils import top1_accuracy, initialize_logger

from codes.attacks.labelflipping import LableFlippingWorker
from codes.attacks.bitflipping import BitFlippingWorker
from codes.attacks.ipm import IPMAttack
from codes.attacks.alittle import ALittleIsEnoughAttack
from codes.attacks.noise import NoiseAttack

from codes.aggregator.coordinatewise_median import CM
from codes.aggregator.clipping import Clipping
from codes.aggregator.clippedclustering import ClusteringClipping
from codes.aggregator.clustering import Clustering
from codes.aggregator.rfa import RFA
from codes.aggregator.trimmed_mean import TM
from codes.aggregator.krum import Krum
from codes.aggregator.base import Mean
from codes.aggregator.autogm import AutoGM
from codes.cctnets import cct_2_3x2_32



options = parse_arguments()
EXP_ID = os.path.basename(__file__)[:-3]  # the file name only
ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) + "/../"
DATA_PATH = os.path.join(ROOT_DIR, "../data/cifar10/data_cache" + (".obj" if options.iid else "_alpha0.1.obj"))
DATA_DIR = os.path.join(ROOT_DIR, "../data/cifar10/")
EXP_DIR = os.path.join(ROOT_DIR, f"outputs/{EXP_ID}"
                       + ("_fedavg/" if options.fedavg else "/"))
# EXP_DIR = os.path.join(ROOT_DIR, f"outputs/{EXP_ID}/")


N_WORKERS = options.num_workers
N_BYZ = options.num_byzantine
BATCH_SIZE = options.batch_size
TEST_BATCH_SIZE = 128
MAX_BATCHES_PER_EPOCH = 50
EPOCHS = options.round
LR = options.lr
MOMENTUM = options.momentum

# LOG_DIR = EXP_DIR + "log"
LOG_DIR = (
        EXP_DIR
        + ("debug/" if options.debug else "")
        + f"f{N_BYZ}_{options.attack}_{options.agg}_m{options.momentum}"
        + (f"_lr{options.lr}" if options.lr != 0.1 else "")
        + (f"_bz{options.batch_size}" if options.batch_size != 32 else "")
        + f"_seed{options.seed}"
)


def get_sampler_callback(rank):
    def sampler_callback(x):
        return DistributedSampler(
            num_replicas=N_WORKERS - N_BYZ,
            rank=rank if rank < N_BYZ else rank - N_BYZ,
            shuffle=True,
            dataset=x,
        )
    
    return sampler_callback


def _get_aggregator():
    if options.agg == "avg":
        return Mean()
    
    if options.agg == "cm":
        return CM()
    
    if options.agg == "cp":
        return Clipping(tau=100, n_iter=1)
    
    if options.agg == "rfa":
        return RFA()
    
    if options.agg == "tm":
        return TM(b=N_BYZ)
    
    if options.agg == "krum":
        return Krum(n=N_WORKERS, f=N_BYZ, m=1)
    
    if options.agg == "clippedclustering":
        return ClusteringClipping()
    
    if options.agg == "clustering":
        return Clustering()
    
    if options.agg == 'autogm':
        return AutoGM()
    raise NotImplementedError(options.agg)


def initialize_worker(
        trainer, worker_rank, model, optimizer, loss_func, device, is_fedavg=False, kwargs=None):
    train_loader = cifar10(
        data_dir=DATA_DIR,
        data_path=DATA_PATH,
        train=True,
        download=True,
        batch_size=BATCH_SIZE,
        # sampler_callback=get_sampler_callback(worker_rank),
        dataset_cls=datasets.CIFAR10,
        drop_last=True,  # Exclude the influence of non-full batch.
        worker_rank=worker_rank,
        **kwargs,
    )
    # NOTE: The first N_BYZ nodes are Byzantine
    if worker_rank < N_BYZ:
        if options.attack == "BF":
            return BitFlippingWorker(
                data_loader=train_loader,
                model=model,
                loss_func=loss_func,
                device=device,
                optimizer=optimizer,
                **kwargs,
            )
        
        if options.attack == "LF":
            return LableFlippingWorker(
                revertible_label_transformer=lambda target: 9 - target,
                data_loader=train_loader,
                model=model,
                loss_func=loss_func,
                device=device,
                optimizer=optimizer,
                **kwargs,
            )
        
        if options.attack == "IPM":
            attacker = IPMAttack.remote(
                epsilon=0.5,
                is_fedavg=is_fedavg,
                data_loader=train_loader,
                model=model,
                loss_func=loss_func,
                device=device,
                optimizer=optimizer,
                **kwargs,
            )
            attacker.configure.remote(trainer)
            return attacker
        
        if options.attack == "IPM_large":
            attacker = IPMAttack(
                epsilon=1000.0,
                is_fedavg=is_fedavg,
                data_loader=train_loader,
                model=model,
                loss_func=loss_func,
                device=device,
                optimizer=optimizer,
                **kwargs,
            )
            attacker.configure(trainer)
            return attacker
        
        if options.attack == "ALIE":
            attacker = ALittleIsEnoughAttack(
                n=N_WORKERS,
                m=N_BYZ,
                is_fedavg=is_fedavg,
                data_loader=train_loader,
                model=model,
                loss_func=loss_func,
                device=device,
                optimizer=optimizer,
                **kwargs,
            )
            attacker.configure(trainer)
            return attacker
        
        if options.attack == "Noise":
            attacker = NoiseAttack(
                is_fedavg=is_fedavg,
                data_loader=train_loader,
                model=model,
                loss_func=loss_func,
                device=device,
                optimizer=optimizer,
                **kwargs,
            )
            attacker.configure(trainer)
            return attacker
        
        raise NotImplementedError(f"No such attack {options.attack}")
    if options.fedavg:
        return TorchWorker(data_loader=train_loader, model=model, loss_func=loss_func, device=device,
                           optimizer=optimizer, **kwargs, )
    else:
        return WorkerWithMomentum.remote(momentum=MOMENTUM, data_loader=train_loader, model=model, loss_func=loss_func,
                                  device=device, optimizer=optimizer, **kwargs, )


def main(args):
    initialize_logger(LOG_DIR)
    
    if args.use_cuda and not torch.cuda.is_available():
        print("=> There is no cuda device!!!!")
        device = "cpu"
    else:
        device = torch.device("cuda" if args.use_cuda else "cpu")
    
    kwargs = {"pin_memory": True} if args.use_cuda else {}
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # model = get_resnet20(use_cuda=args.use_cuda, gn=False).to(device)
    model = cct_2_3x2_32().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)
    loss_func = CrossEntropyLoss().to(device)
    
    metrics = {"top1": top1_accuracy}
    
    server_opt = torch.optim.SGD(model.parameters(), lr=LR)
    server = TorchServer(server_opt, model=model)
    
    trainer = ParallelTrainer(
        server=server,
        aggregator=_get_aggregator(),
        pre_batch_hooks=[],
        post_batch_hooks=[],
        max_batches_per_epoch=MAX_BATCHES_PER_EPOCH,
        log_interval=args.log_interval,
        metrics=metrics,
        use_cuda=args.use_cuda,
        debug=False,
    )
    
    test_loader = cifar10(
        data_dir=DATA_DIR,
        data_path=DATA_PATH,
        train=False,
        download=True,
        batch_size=TEST_BATCH_SIZE,
        shuffle=False,
        worker_rank=None,
        **kwargs,
    )
    
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        server_opt, milestones=[75, 100], gamma=0.5
    )
    
    evaluator = DistributedEvaluator(
        model=model,
        data_loader=test_loader,
        loss_func=loss_func,
        device=device,
        metrics=metrics,
        use_cuda=args.use_cuda,
        debug=False,
    )
    
    for worker_rank in range(N_WORKERS):
        worker = initialize_worker(
            trainer=trainer,
            worker_rank=worker_rank,
            model=model,
            optimizer=optimizer,
            loss_func=loss_func,
            device=device,
            is_fedavg=args.fedavg,
            kwargs={},
        )
        trainer.add_worker(worker)
    
    # if args.fedavg:
    trainer.parallel_call(lambda worker: worker.detach_model.remote())
    
    # torch.save(model.state_dict(), '../saved_init_model.pt')
    for epoch in range(1, EPOCHS + 1):
        if args.fedavg:
            trainer.train_fedavg(epoch)
        else:
            trainer.train(epoch)
        evaluator.evaluate(epoch)
        scheduler.step()
        print(f"E={epoch}; Learning rate = {scheduler.get_last_lr()[0]:}")

    # torch.save(model.state_dict(), '../saved_final_model.pt')

if __name__ == "__main__":
    if not ray.is_initialized():
        # ray.init(local_mode=True, include_dashboard=True, num_gpus=0)
        ray.init(include_dashboard=True, num_gpus=0)
    main(options)
