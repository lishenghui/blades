import argparse
import os

import torch

from tasks.data_utils import read_data


def parse_arguments():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--use-cuda", action="store_true", default=False)
    parser.add_argument("--fedavg", action="store_true", default=True)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--round", type=int, default=400)
    parser.add_argument("--local_round", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--test_batch_size", type=int, default=128)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--attack", type=str, default='noise', help="Select attack types.")
    parser.add_argument("--dataset", type=str, default='cifar10', help="Dataset")
    parser.add_argument("--agg", type=str, default='clippedclustering', help="Aggregator.")
    parser.add_argument("--momentum", type=float, default=0, help="momentum")
    parser.add_argument("--clipping_tau", type=float, default=100, help="Threshold for clipping")
    parser.add_argument("--lr", type=float, default=0.1, help="learning rate")
    parser.add_argument("--inner-iterations", type=int, default=1, help="[HP]: number of inner iterations.")
    # parser.add_argument("--num_workers", type=int, default=4, help="Number of workers.")
    parser.add_argument("--num_actor", type=int, default=4)
    parser.add_argument("--num_byzantine", type=int, default=1)
    parser.add_argument("--num_gpus", type=int, default=4)
    flag_parser = parser.add_mutually_exclusive_group(required=False)
    flag_parser.add_argument('--iid', dest='iid', action='store_true')
    flag_parser.add_argument('--noniid', dest='iid', action='store_false')
    parser.set_defaults(iid=True)
    options = parser.parse_args()

    if not torch.cuda.is_available():
        print('Unfortunaly, we currently do not have any GPU on your machine. ')
        options.num_gpus = 0
        options.gpu_per_actor = 0
    else:
        options.gpu_per_actor = (options.num_gpus - 0.05) / options.num_gpus
    options.use_cuda = torch.cuda.is_available()
    # if not torch.cuda.is_available():
    #     options.gpu_per_actor = 0
    #     GPU_PER_ACTOR = 0
    # else:
    #     options.gpu_per_actor = (options.num_gpus - 0.05) / options.num_workers
    #     GPU_PER_ACTOR = (options.num_gpus - 0.05) / options.num_workers
    
    EXP_ID = os.path.basename(__file__)[:-3]  # the file name only
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    options.data_path = os.path.join(ROOT_DIR, "../data/cifar10/data_cache" + (".obj" if options.iid else "_alpha0.1.obj"))
    options.data_dir = os.path.join(ROOT_DIR, "../data/cifar10/")
    EXP_DIR = os.path.join(ROOT_DIR, f"outputs/{options.dataset}"
                           + ("_fedavg/" if options.fedavg else "/"))
    # EXP_DIR = os.path.join(ROOT_DIR, f"outputs/{EXP_ID}/")

    # LOG_DIR = EXP_DIR + "log"
    options.log_dir = (
            EXP_DIR
            + ("debug/" if options.debug else "")
            + f"f{options.num_byzantine}_{options.attack}_{options.agg}_m{options.momentum}"
            + (f"_lr{options.lr}" if options.lr != 0.1 else "")
            + (f"_bz{options.batch_size}" if options.batch_size != 32 else "")
            + f"_seed{options.seed}"
    )
    _, _, train_data, _ = read_data(data_path=options.data_path)
    options.num_clients = len(list(train_data.keys()))
    
    return options
