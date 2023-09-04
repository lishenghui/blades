# import os
import ray
import torch
from ray.runtime_env import RuntimeEnv

from fllib.communication import Communicator
from fllib.constants import MINIMUM_GPU_FRACTION

runtime_env = RuntimeEnv(
    # env_vars={"CUDA_VISIBLE_DEVICES": "0,1,2,3"},
    eager_install=False
)


# def creat_communicator_group(world_size=2):


def test_basic_communication():
    size = (1, 10)
    cuda_visible_devices = "0,1,2,3"
    print(cuda_visible_devices)
    cli_comm = Communicator.as_remote(
        num_gpus=MINIMUM_GPU_FRACTION, runtime_env=runtime_env
    ).remote(
        world_size=2, world_rank=1, cuda=0, cuda_visible_devices=cuda_visible_devices
    )
    svr_comm = Communicator.as_remote(
        num_gpus=MINIMUM_GPU_FRACTION, runtime_env=runtime_env
    ).remote(
        world_size=2,
        world_rank=0,
        pkg_size=size,
        buffer_len=5,
        cuda=1,
        cuda_visible_devices=cuda_visible_devices,
    )
    msg_queue = ray.get(cli_comm.get_msg_queue.remote())

    cli_device = ray.get(cli_comm.get_device.remote())
    tensor1 = 1 * torch.ones(size=size).to(cli_device)
    tensor2 = 2 * torch.ones(size=size).to(cli_device)

    tensors = [tensor1, tensor2]
    num_com = len(tensors)

    for tensor in tensors:
        msg_queue.put(tensor)

    ray.get(
        [cli_comm.local_round.remote(num_com), svr_comm.local_round.remote(num_com)]
    )

    server_buffer = ray.get(svr_comm.get_buffers.remote())
    assert len(tensors) == len(server_buffer)
    assert all([torch.equal(a, b.to(a.device)) for a, b in zip(tensors, server_buffer)])


if __name__ == "__main__":
    ray.init()
    test_basic_communication()
