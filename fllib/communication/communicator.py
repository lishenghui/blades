import os
from typing import (
    Any,
    Tuple,
    Union,
)

import ray
import torch
import torch.distributed as dist
from torch.multiprocessing.reductions import reduce_tensor

# from fllib.utils.annotations import DeveloperAPI
from fllib.constants import MINIMUM_GPU_FRACTION, SERVER_RANK, DEFAULT_MASTER_PORT
from fllib.utils.queue import Queue


class _BufferList:
    def __init__(
        self,
        package_size,
        length: int,
        device="cuda:0",
        buffers=None,
        actual_indexes=None,
    ) -> None:
        self.device = device
        self._dummy_tensor = torch.full(package_size, torch.nan).to(device)

        if buffers is not None:
            self._buffer_list = [
                _BufferList._reconstruct_tensor(meta) for meta in buffers
            ]
        else:
            self._buffer_list = [
                torch.full(package_size, torch.nan).to(device) for _ in range(length)
            ]
        if actual_indexes is None:
            self._actual_indexes = torch.zeros(length, dtype=torch.bool).to(device)
        else:
            self._actual_indexes = _BufferList._reconstruct_tensor(actual_indexes)

        self._buffer_base = 0

    @staticmethod
    def _reconstruct_tensor(meta):
        return meta[0](*meta[1])

    def _serialization_helper(self):
        """This is defined in order to make pickling work.

        Returns:
            A dictionary of the information needed to reconstruct the object.
        """
        actual_indexes_meta = reduce_tensor(self._actual_indexes)
        buffer_list_meta = [reduce_tensor(tensor) for tensor in self._buffer_list]
        state = {
            "package_size": self.dummy_tensor.size(),
            "length": len(self._actual_indexes),
            "actual_ind": actual_indexes_meta,
            "buffer_list": buffer_list_meta,
        }

        return state

    @classmethod
    def _deserialization_helper(cls, state):
        """This is defined in order to make pickling work.

        Args:
            state: The serialized state of the actor handle.
        """
        return state
        # return cls(
        #     package_size=state["package_size"],
        #     length=state["length"],
        #     buffers=state["buffer_list"],
        #     actual_indexes=state["actual_ind"],
        # )

    def __reduce__(self) -> tuple[Any, ...]:
        meta_state = self._serialization_helper()
        return _BufferList._deserialization_helper, (meta_state,)

    @property
    def dummy_tensor(self):
        return self._dummy_tensor

    @property
    def actual_tensors(self):
        return [i for (i, v) in zip(self._buffer_list, self._actual_indexes) if v]

    def get_next_empty_list(self, length, with_dummy=False):
        if with_dummy:
            length -= 1
            pre = [self._dummy_tensor]
        else:
            pre = []

        slice_ = slice(self._buffer_base, self._buffer_base + length, 1)
        empty_list = self._buffer_list[slice_]

        self._actual_indexes[self._buffer_base : self._buffer_base + length].copy_(
            torch.ones(length)
        )

        empty_list = pre + empty_list
        self._buffer_base += length
        return empty_list

    def reset(self):
        import torch

        self._buffer_base = 0
        self._actual_indexes.copy_(torch.zeros_like(self._actual_indexes))


class Communicator:
    def setup(
        self,
        world_size: int = None,
        world_rank: int = None,
        pkg_size: Tuple = None,
        buffer_len: int = 1,
        cuda: int = None,
        cuda_visible_devices: Union[Tuple[int], str] = None,
    ) -> None:
        if cuda is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = (
                cuda_visible_devices
                if isinstance(cuda_visible_devices, str)
                else ",".join(cuda_visible_devices)
            )
            self._device = f"cuda:{cuda}"
        else:
            self._device = "cpu"

        self._msg_queue = Queue(actor_options={"num_gpus": MINIMUM_GPU_FRACTION})

        if pkg_size is None:
            pkg_size = (1, 1)

        print("device: ", self.device)
        self._buffer = _BufferList(pkg_size, buffer_len, device=self.device)

        self._gather_handles = []

        master_addr = "147.8.183.198"
        self.process_group = self._setup_dist(
            world_size=world_size,
            world_rank=world_rank,
            node_rank=0 if ray._private.worker.global_worker.node.is_head() else 1,
            master_addr=master_addr,
            master_port=DEFAULT_MASTER_PORT,
            group_name="default",
            backend="nccl",
        )

    @staticmethod
    def _setup_dist(
        world_size: int,
        world_rank: int,
        node_rank: int,
        master_addr: str,
        master_port: str,
        group_name: str,
        backend: str,
    ):
        """Initialize the distributed environment."""

        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = master_port
        os.environ["NODE_RANK"] = str(node_rank)
        import torch.distributed as dist

        print("world_size: ", world_size)
        group = dist.init_process_group(
            backend=backend,
            group_name=group_name,
            rank=world_rank,
            world_size=world_size,
        )

        return group

    @property
    def world_size(self):
        return dist.get_world_size()

    @property
    def rank(self):
        return dist.get_rank()

    @property
    def device(self):
        if not hasattr(self, "_device") or self._device is None:
            self._device = "cpu"
        else:
            return self._device

    def get_msg_queue(self):
        return self._msg_queue

    def get_buffers(self):
        return self._buffer.actual_tensors

    def local_round(self, num_steps=1):
        self._buffer.reset()
        for step in range(num_steps):
            self.gather_async()
        self._wait_for_gather()

    def gather_async(self):
        if self.rank == SERVER_RANK:
            dummy_tensor = self._buffer.dummy_tensor
            tensor_list = self._buffer.get_next_empty_list(
                self.world_size, with_dummy=True
            )
            self._gather_handles.append(
                dist.gather(dummy_tensor, tensor_list, async_op=True).get_future()
            )
        else:
            tensor = self._msg_queue.get()
            self._gather_handles.append(dist.gather(tensor, async_op=True))

    def broadcast_tensor(self, tensor, root=0):
        dist.broadcast(tensor, src=root)

    def _wait_for_gather(self):
        # Make sure that the calls are done before moving out.
        _ = list(map(lambda x: x.wait(), self._gather_handles))
        self._gather_handles = []

    def get_device(self):
        return self.device
