from typing import Union, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import linalg as LA
from torch._six import inf

_tensor_or_tensors = Union[torch.Tensor, Iterable[torch.Tensor]]


class HLoss(nn.Module):
    def __init__(self):
        super(HLoss, self).__init__()
    
    def forward(self, x):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = -1.0 * b.sum()
        return b


def l2dist(model1, model2):
    return LA.norm(torch.tensor([LA.norm(model1[k] - model2[k]) for k in model1 if model1[k].dtype != torch.int64]))


def l2norm(model):
    return LA.norm(torch.tensor([LA.norm(model[k]) for k in model if model[k].dtype != torch.int64]))


def cos_sim(model1, model2):
    return torch.sum(torch.tensor([torch.sum(model1[k] * model2[k]) for k in model1])) / torch.max(
        l2norm(model1) * l2norm(model2), torch.tensor(0.00001))


def clip_para_norm_(
        parameters: _tensor_or_tensors, max_norm: float, norm_type: float = 2.0,
        error_if_nonfinite: bool = False) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters.values()]
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].device
    if norm_type == inf:
        norms = [p.detach().abs().max().to(device) for p in parameters]
        total_norm = norms[0] if len(norms) == 1 else torch.max(torch.stack(norms))
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.detach(), norm_type).to(device) for p in parameters
                                             if p.dtype != torch.int64]), norm_type)
    if error_if_nonfinite and torch.logical_or(total_norm.isnan(), total_norm.isinf()):
        raise RuntimeError(
            f'The total norm of order {norm_type} for gradients from '
            '`parameters` is non-finite, so it cannot be clipped. To disable '
            'this error and scale the gradients by the non-finite norm anyway, '
            'set `error_if_nonfinite=False`')
    clip_coef = max_norm / (total_norm + 1e-6)
    # Note: multiplying by the clamped coef is redundant when the coef is clamped to 1, but doing so
    # avoids a `if clip_coef < 1:` conditional which can require a CPU <=> device synchronization
    # when the gradients do not reside in CPU memory.
    clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
    for p in parameters:
        if p.dtype != torch.int64:
            p.detach().mul_(clip_coef_clamped.to(p.device))
    return total_norm


def clip_tensor_norm_(
        parameters: _tensor_or_tensors, max_norm: float, norm_type: float = 2.0,
        error_if_nonfinite: bool = False) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].device
    if norm_type == inf:
        norms = [p.detach().abs().max().to(device) for p in parameters]
        total_norm = norms[0] if len(norms) == 1 else torch.max(torch.stack(norms))
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.detach(), norm_type).to(device) for p in parameters
                                             if p.dtype != torch.int64]), norm_type)
    if error_if_nonfinite and torch.logical_or(total_norm.isnan(), total_norm.isinf()):
        raise RuntimeError(
            f'The total norm of order {norm_type} for gradients from '
            '`parameters` is non-finite, so it cannot be clipped. To disable '
            'this error and scale the gradients by the non-finite norm anyway, '
            'set `error_if_nonfinite=False`')
    clip_coef = max_norm / (total_norm + 1e-6)
    # Note: multiplying by the clamped coef is redundant when the coef is clamped to 1, but doing so
    # avoids a `if clip_coef < 1:` conditional which can require a CPU <=> device synchronization
    # when the gradients do not reside in CPU memory.
    clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
    for p in parameters:
        if p.dtype != torch.int64:
            return p.detach().mul_(clip_coef_clamped.to(p.device))
