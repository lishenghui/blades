import logging
import os
import random
import shutil
from importlib import reload

import numpy as np
import ruamel.yaml as yaml
import torch
from torch import nn


class BColors(object):
    HEADER = "\033[95m"
    OK_BLUE = "\033[94m"
    OK_CYAN = "\033[96m"
    OK_GREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    END_C = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def touch(fname: str, times=None, create_dirs: bool = False):
    if create_dirs:
        base_dir = os.path.dirname(fname)
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
    with open(fname, "a"):
        os.utime(fname, times)


def touch_dir(base_dir: str) -> None:
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k."""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def top1_accuracy(output, target):
    return accuracy(output, target, topk=(1,))[0].item()


def log(*args, **kwargs):
    pass


def log_dict(*args, **kwargs):
    pass


def initialize_logger(log_root):
    logging.shutdown()
    reload(logging)
    if not os.path.exists(log_root):
        os.makedirs(log_root)
    else:
        shutil.rmtree(log_root)
        os.makedirs(log_root)

    print(f"Logging files to {log_root}")

    # Only to file; One dict per line; Easy to process
    json_logger = logging.getLogger("stats")
    json_logger.setLevel(logging.INFO)
    fh = logging.FileHandler(os.path.join(log_root, "stats"))
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(message)s"))
    json_logger.addHandler(fh)

    debug_logger = logging.getLogger("debug")
    debug_logger.setLevel(logging.INFO)
    # ch = logging.StreamHandler()
    # ch.setLevel(logging.INFO)
    # ch.setFormatter(logging.Formatter("%(message)s"))
    # debug_logger.addHandler(ch)
    fh = logging.FileHandler(os.path.join(log_root, "debug"))
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(message)s"))
    debug_logger.addHandler(fh)


def reset_model_weights(model: nn.Module) -> None:
    """
    refs:

    - https://discuss.pytorch.org/t/how-to-re-set-alll-parameters-in-a-network/20819/6
    - https://pytorch.org/docs/stable/generated/torch.nn.Module.html
    """

    @torch.no_grad()
    def weight_reset(m: nn.Module):
        # - check if the current module has reset_parameters & if it's callabed
        # called it on m
        reset_parameters = getattr(m, "reset_parameters", None)
        if callable(reset_parameters):
            m.reset_parameters()

    # Applies fn recursively to every submodule see:
    # https://pytorch.org/docs/stable/generated/torch.nn.Module.html
    model.apply(fn=weight_reset)


def set_random_seed(seed_value=0, use_cuda=False):
    np.random.seed(seed_value)  # cpu vars
    random.seed(seed_value)  # Python
    torch.manual_seed(seed_value)
    os.environ["PYTHONHASHSEED"] = str(seed_value)  # Python hash buildin
    if use_cuda:
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def over_write_args_from_file(args, yml):
    """overwrite arguments according to config file."""
    if yml == "":
        return
    with open(yml, "r", encoding="utf-8") as f:
        dic = yaml.load(f.read(), Loader=yaml.Loader)
        for k in dic:
            setattr(args, k, dic[k])
