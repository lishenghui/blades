import os

import torch.distributed as dist


def setup_dist(
    world_size,
    rank,
    addr="127.0.0.1",
    port="7777",
    group_name="default",
    backend="nccl",
):
    """Initialize the distributed environment."""
    os.environ["MASTER_ADDR"] = addr
    os.environ["MASTER_PORT"] = port
    group = dist.init_process_group(
        backend, group_name=group_name, rank=rank, world_size=world_size
    )

    return group
