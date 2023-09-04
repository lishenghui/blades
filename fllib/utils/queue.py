from typing import Optional, Dict, Any

import torch
from ray.util.queue import Queue
from torch.multiprocessing.reductions import reduce_tensor


class Queue(Queue):
    def __init__(self, maxsize: int = 0, actor_options: Optional[Dict] = None) -> None:
        super().__init__(maxsize, actor_options)

    def put(
        self, item: Any, block: bool = True, timeout: Optional[float] = None
    ) -> None:
        if torch.is_tensor(item) and item.is_cuda:
            return super().put(reduce_tensor(item), block, timeout)
        else:
            return super().put(item, block, timeout)

    def get(self, block: bool = True, timeout: Optional[float] = None) -> Any:
        ret = super().get(block, timeout)
        if torch.is_tensor(ret) or not isinstance(ret, tuple):
            return ret
        else:
            return ret[0](*ret[1])

    async def get_async(
        self, block: bool = True, timeout: Optional[float] = None
    ) -> Any:
        return super().get_async(block, timeout)
