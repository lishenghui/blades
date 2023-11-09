import sys
import unittest
from typing import Any, Callable, Optional, TypeVar

import ray

from fllib.core.execution.actor_manager import ActorManager

T = TypeVar("T")


@ray.remote
class Worker:
    def __init__(self, i):
        self.count = i

    def apply(
        self,
        func: Callable[["Worker", Optional[Any], Optional[Any]], T],
        *args,
        **kwargs,
    ) -> T:
        return func(self, *args, **kwargs)


class TestActorManager(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        ray.init()

    @classmethod
    def tearDownClass(cls) -> None:
        ray.shutdown()

    def test_execute_with_actor_pool(self):
        """Test synchronous remote calls to only healthy actors."""
        values = [1, 2, 3, 4]
        actors = [Worker.remote(i) for i in range(4)]
        manager = ActorManager(actors=actors)
        results = manager.execute_with_actor_pool(
            lambda actor, value: value, values=values
        )
        self.assertEquals(results, values)

        affinity_actors = [[actor] for actor in actors]
        results = manager.execute_with_actor_pool(
            lambda actor, value: value, values=values, affinity_actors=affinity_actors
        )
        manager.clear()


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main(["-v", __file__]))
