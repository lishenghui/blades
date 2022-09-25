import argparse
import os

import torch

from blades.utils.utils import over_write_args_from_file


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--non_iid", action="store_true", default=False)
    parser.add_argument("--local_mode", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--global_round", type=int, default=400)
    parser.add_argument("--local_round", type=int, default=50)
    parser.add_argument("--serv_momentum", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--test_batch_size", type=int, default=128)
    parser.add_argument("--validate_interval", type=int, default=100)
    parser.add_argument("--trusted_id", type=int, default=None)
    parser.add_argument(
        "--attack", type=str, default="signflipping", help="Select attack types."
    )
    parser.add_argument("--dataset", type=str, default="cifar10", help="Dataset")
    parser.add_argument(
        "--algorithm",
        type=str,
        default="fedsgd",
        help="Optimization algorithm, either 'fedavg' or 'fedsgd'.",
    )
    parser.add_argument(
        "--agg", type=str, default="clippedclustering", help="Aggregator."
    )
    parser.add_argument("--lr", type=float, default=0.1, help="learning rate")
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--num_clients", type=int, default=20)
    parser.add_argument("--num_byzantine", type=int, default=5)

    parser.add_argument("--num_actors", type=int, default=5)
    parser.add_argument("--gpu_per_actor", type=float, default=0.2)

    parser.add_argument("--dp", action="store_true", default=False)

    # Parameters for DP. They will take effect only if `dp`
    # is  `True`.
    parser.add_argument("--dp_privacy_delta", type=float, default=1e-6)
    parser.add_argument("--dp_privacy_epsilon", type=float, default=1.0)
    parser.add_argument("--dp_clip_threshold", type=float, default=0.5)

    parser.add_argument(
        "--config_path", type=str, default=None, help="Path to config file."
    )

    options = parser.parse_args()
    options.agg = options.agg.lower()
    options.attack = options.attack.lower()
    if options.algorithm == "fedsgd":
        options.local_round = 1
    options.dp_privacy_sensitivity = 2 * options.dp_clip_threshold / options.batch_size
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

    EXP_DIR = os.path.join(ROOT_DIR, f"outputs/{options.dataset}")

    attack_args = {
        "signflipping": {},
        "noise": {"std": 0.1},
        "labelflipping": {},
        "permutation": {},
        "attackclippedclustering": {},
        "fangattack": {},
        "distancemaximization": {},
        "ipm": {"epsilon": 0.5},
        "alie": {
            "num_clients": options.num_clients,
            "num_byzantine": options.num_byzantine,
        },
    }
    agg_args = {
        "trimmedmean": {"num_excluded": options.num_byzantine},
        "median": {},
        "mean": {},
        "signguard": {},
        "geomed": {},
        "dnc": {"num_byzantine": options.num_byzantine},
        "autogm": {"lamb": 2.0},
        "clippedclustering": {"max_tau": 2.0, "signguard": True, "linkage": "average"},
        "clustering": {},
        "centeredclipping": {},
        "multikrum": {"num_excluded": options.num_byzantine, "k": 5},
    }

    options.attack_kws = attack_args[options.attack]
    options.aggregator_kws = agg_args[options.agg]

    options.adversary_args = {
        "fangattack": {"num_byzantine": options.num_byzantine, "agg": "median"},
        "distancemaximization": {
            "num_byzantine": options.num_byzantine,
            "agg": "trimmedmean",
        },
    }
    if options.config_path:
        options.aggregator_kws = {}
        over_write_args_from_file(options, options.config_path)
        options.agg = options.agg.lower()
        options.attack = options.attack.lower()

    options.log_dir = (
        EXP_DIR
        + f"_{options.algorithm}"
        + f"/b{options.num_byzantine}"
        + f"_{options.attack}"
        + (
            "_" + "_".join([k + str(v) for k, v in options.attack_kws.items()])
            if options.attack_kws
            else ""
        )
        + f"_{options.agg}"
        + (
            "_" + "_".join([k + str(v) for k, v in options.aggregator_kws.items()])
            if options.aggregator_kws
            else ""
        )
        + (f"_lr{options.lr}")
        + (f"_serv_momentum{options.serv_momentum}")
        + (f"_bz{options.batch_size}")
        + (f"_localround{options.local_round}")
        + ("_noniid" if options.non_iid else "")
        + (
            f"_privacy_epsilon{options.dp_privacy_epsilon}_clip_threshold"
            f"{options.dp_clip_threshold}"
            if options.dp
            else ""
        )
        + f"_seed{options.seed}"
    )

    if not torch.cuda.is_available():
        print("We currently do not have any GPU on your machine. ")
        options.num_gpus = 0
        options.gpu_per_actor = 0
    return options


options = parse_arguments()
