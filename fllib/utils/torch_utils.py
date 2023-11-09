import logging
import os
from typing import Iterable, Dict, Optional, TYPE_CHECKING, Union

import numpy as np
import ray

from ray.rllib.utils.annotations import Deprecated, PublicAPI
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import (
    LocalOptimizer,
    TensorStructType,
    TensorType,
)
from torch import inf

# import tree  # pip install dm_tree

if TYPE_CHECKING:
    from ray.rllib.policy.torch_policy import TorchPolicy

logger = logging.getLogger(__name__)
torch, nn = try_import_torch()

# Limit values suitable for use as close to a -inf logit. These are useful
# since -inf / inf cause NaNs during backprop.
FLOAT_MIN = -3.4e38
FLOAT_MAX = 3.4e38

_tensor_or_tensors = Union[torch.Tensor, Iterable[torch.Tensor]]


@PublicAPI
def apply_grad_clipping(
    policy: "TorchPolicy", optimizer: LocalOptimizer, loss: TensorType
) -> Dict[str, TensorType]:
    """Applies gradient clipping to already computed grads inside `optimizer`.

    Args:
        policy: The TorchPolicy, which calculated `loss`.
        optimizer: A local torch optimizer object.
        loss: The torch loss tensor.

    Returns:
        An info dict containing the "grad_norm" key and the resulting clipped
        gradients.
    """
    grad_gnorm = 0
    if policy.config["grad_clip"] is not None:
        clip_value = policy.config["grad_clip"]
    else:
        clip_value = np.inf

    for param_group in optimizer.param_groups:
        # Make sure we only pass params with grad != None into torch
        # clip_grad_norm_. Would fail otherwise.
        params = list(filter(lambda p: p.grad is not None, param_group["params"]))
        if params:
            # PyTorch clips gradients inplace and returns the norm before clipping
            # We therefore need to compute grad_gnorm further down (fixes #4965)
            global_norm = nn.utils.clip_grad_norm_(params, clip_value)

            if isinstance(global_norm, torch.Tensor):
                global_norm = global_norm.cpu().numpy()

            grad_gnorm += min(global_norm, clip_value)

    if grad_gnorm > 0:
        return {"grad_gnorm": grad_gnorm}
    else:
        # No grads available
        return {}


@Deprecated(new="ray/rllib/utils/numpy.py::convert_to_numpy", error=True)
def convert_to_non_torch_type(stats: TensorStructType) -> TensorStructType:
    pass


@PublicAPI
def get_device(config):
    """Returns a torch device edepending on a config and current worker
    index."""

    # Figure out the number of GPUs to use on the local side (index=0) or on
    # the remote workers (index > 0).
    worker_idx = config.get("worker_index", 0)
    if (
        not config["_fake_gpus"]
        and ray._private.worker._mode() == ray._private.worker.LOCAL_MODE
    ):
        num_gpus = 0
    elif worker_idx == 0:
        num_gpus = config["num_gpus"]
    else:
        num_gpus = config["num_gpus_per_worker"]
    # All GPU IDs, if any.
    gpu_ids = list(range(torch.cuda.device_count()))

    # Place on one or more CPU(s) when either:
    # - Fake GPU mode.
    # - num_gpus=0 (either set by user or we are in local_mode=True).
    # - No GPUs available.
    if config["_fake_gpus"] or num_gpus == 0 or not gpu_ids:
        return torch.device("cpu")
    # Place on one or more actual GPU(s), when:
    # - num_gpus > 0 (set by user) AND
    # - local_mode=False AND
    # - actual GPUs available AND
    # - non-fake GPU mode.
    else:
        # We are a remote worker (WORKER_MODE=1):
        # GPUs should be assigned to us by ray.
        if ray._private.worker._mode() == ray._private.worker.WORKER_MODE:
            gpu_ids = ray.get_gpu_ids()

        if len(gpu_ids) < num_gpus:
            raise ValueError(
                "TorchPolicy was not able to find enough GPU IDs! Found "
                f"{gpu_ids}, but num_gpus={num_gpus}."
            )
        return torch.device("cuda")


def vector_to_parameters(vec: torch.Tensor, parameters: Iterable[torch.Tensor]) -> None:
    r"""Convert one vector to the parameters.

    Args:
        vec (Tensor): a single vector represents the parameters of a model.
        parameters (Iterable[Tensor]): an iterator of Tensors that are the
            parameters of a model.
    """
    # Ensure vec of type Tensor
    if not isinstance(vec, torch.Tensor):
        raise TypeError(
            "expected torch.Tensor, but got: {}".format(torch.typename(vec))
        )
    # Flag for the _device where the parameter is located
    param_device = None

    # Pointer for slicing the vector for each parameter
    pointer = 0
    for param in parameters:
        if not param.requires_grad:
            continue

        # Ensure the parameters are located in the same _device
        param_device = _check_param_device(param, param_device)

        # The length of the parameter
        num_param = param.numel()
        # Slice the vector, reshape it, and replace the old data of the parameter
        param.data = vec[pointer : pointer + num_param].view_as(param).data

        # Increment the pointer
        pointer += num_param


@PublicAPI
def set_torch_seed(seed: Optional[int] = None) -> None:
    """Sets the torch random seed to the given value.

    Args:
        seed: The seed to use or None for no seeding.
    """
    if seed is not None and torch:
        torch.manual_seed(seed)
        # See https://github.com/pytorch/pytorch/issues/47672.
        cuda_version = torch.version.cuda
        if cuda_version is not None and float(torch.version.cuda) >= 10.2:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = "4096:8"
        else:
            # Not all Operations support this.
            torch.use_deterministic_algorithms(True)
        # This is only for Convolution no problem.
        torch.backends.cudnn.deterministic = True


def parameters_to_vector(parameters: Iterable[torch.Tensor]) -> torch.Tensor:
    r"""Convert parameters to one vector.

    Args:
        parameters (Iterable[Tensor]): an iterator of Tensors that are the
            parameters of a model.
    Returns:
        The parameters represented by a single vector
    """
    # Flag for the _device where the parameter is located
    param_device = None

    vec = []
    for param in parameters:
        if not param.requires_grad:
            continue
        # Ensure the parameters are located in the same _device
        param_device = _check_param_device(param, param_device)

        vec.append(param.view(-1))
    return torch.cat(vec)


def _check_param_device(param: torch.Tensor, old_param_device: Optional[int]) -> int:
    r"""This helper function is to check if the parameters are located in the
    same _device.

    Currently, the conversion between model parameters and single
    vector form is not supported for multiple allocations, e.g. parameters in
    different GPUs, or mixture of CPU/GPU.
    Args:
        param ([Tensor]): a Tensor of a parameter of a model
        old_param_device (int): the _device where the first parameter of a
                                model is allocated.
    Returns:
        old_param_device (int): report _device for the first time
    """

    # Meet the first parameter
    if old_param_device is None:
        old_param_device = param.get_device() if param.is_cuda else -1
    else:
        warn = False
        if param.is_cuda:  # Check if in same GPU
            warn = param.get_device() != old_param_device
        else:  # Check if in CPU
            warn = old_param_device != -1
        if warn:
            raise TypeError(
                "Found two parameters on different devices, "
                "this is currently not supported."
            )
    return old_param_device


def clip_para_norm_(
    parameters: _tensor_or_tensors,
    max_norm: float,
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters.values()]
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.0)
    device = parameters[0].device
    if norm_type == inf:
        norms = [p.detach().abs().max().to(device) for p in parameters]
        total_norm = norms[0] if len(norms) == 1 else torch.max(torch.stack(norms))
    else:
        total_norm = torch.norm(
            torch.stack(
                [
                    torch.norm(p.detach(), norm_type).to(device)
                    for p in parameters
                    if p.dtype != torch.int64
                ]
            ),
            norm_type,
        )
    if error_if_nonfinite and torch.logical_or(total_norm.isnan(), total_norm.isinf()):
        raise RuntimeError(
            f"The total norm of order {norm_type} for gradients from "
            "`parameters` is non-finite, so it cannot be clipped. To disable "
            "this error and scale the gradients by the non-finite norm anyway, "
            "set `error_if_nonfinite=False`"
        )
    clip_coef = max_norm / (total_norm + 1e-6)
    # Note: multiplying by the clamped coef is redundant when the coef is clamped to 1,
    # but doing so avoids a `if clip_coef < 1:` conditional which can require a CPU <=>
    # device synchronization when the gradients do not reside in CPU memory.
    clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
    for p in parameters:
        if p.dtype != torch.int64:
            p.detach().mul_(clip_coef_clamped.to(p.device))
    return total_norm


def clip_tensor_norm_(
    parameters: _tensor_or_tensors,
    max_norm: float,
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.0)
    device = parameters[0].device
    if norm_type == inf:
        norms = [p.detach().abs().max().to(device) for p in parameters]
        total_norm = norms[0] if len(norms) == 1 else torch.max(torch.stack(norms))
    else:
        total_norm = torch.norm(
            torch.stack(
                [
                    torch.norm(p.detach(), norm_type).to(device)
                    for p in parameters
                    if p.dtype != torch.int64
                ]
            ),
            norm_type,
        )
    if error_if_nonfinite and torch.logical_or(total_norm.isnan(), total_norm.isinf()):
        raise RuntimeError(
            f"The total norm of order {norm_type} for gradients from "
            "`parameters` is non-finite, so it cannot be clipped. To disable "
            "this error and scale the gradients by the non-finite norm anyway, "
            "set `error_if_nonfinite=False`"
        )
    clip_coef = max_norm / (total_norm + 1e-6)
    # Note: multiplying by the clamped coef is redundant when the coef is clamped to 1,
    # but doing so avoids a `if clip_coef < 1:` conditional which can require a
    # CPU <=> device synchronization when the gradients do not reside in CPU memory.
    clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
    for p in parameters:
        if p.dtype != torch.int64:
            return p.detach().mul_(clip_coef_clamped.to(p.device))
