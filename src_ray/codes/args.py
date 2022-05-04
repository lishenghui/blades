import argparse
import torch


NUM_GPUS = 4
NUM_WORKERS = 20
if not torch.cuda.is_available():
    print('Unfortunaly, we currently do not have any GPU on your machine. ')
    NUM_GPUS = 0
    GPU_PER_ACTOR = 0
else:
    GPU_PER_ACTOR = (NUM_GPUS - 0.0) / NUM_WORKERS

def parse_arguments():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--use-cuda", action="store_true", default=False)
    parser.add_argument("--fedavg", action="store_true", default=False)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--round", type=int, default=400)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--attack", type=str, default='IPM', help="Select from BF and LF.")
    parser.add_argument("--agg", type=str, default='avg', help="Aggregator.")
    parser.add_argument("--momentum", type=float, default=0, help="momentum")
    parser.add_argument("--lr", type=float, default=0.1, help="learning rate")
    parser.add_argument("--inner-iterations", type=int, default=1, help="[HP]: number of inner iterations.")
    # parser.add_argument("--num_workers", type=int, default=20, help="Number of workers.")
    parser.add_argument("--num_byzantine", type=int, default=0)
    flag_parser = parser.add_mutually_exclusive_group(required=False)
    flag_parser.add_argument('--iid', dest='iid', action='store_true')
    flag_parser.add_argument('--noniid', dest='iid', action='store_false')
    parser.set_defaults(iid=True)
    options = parser.parse_args()

    options.num_workers = NUM_WORKERS
    options.use_cuda = torch.cuda.is_available()
    # if not torch.cuda.is_available():
    #     options.gpu_per_actor = 0
    #     GPU_PER_ACTOR = 0
    # else:
    #     options.gpu_per_actor = (options.num_gpus - 0.05) / options.num_workers
    #     GPU_PER_ACTOR = (options.num_gpus - 0.05) / options.num_workers
    return options