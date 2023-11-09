import unittest

import ray

from fllib.core.execution.scaling_config import WorkerGroupScalingConfig
from fllib.core.execution.worker import Worker


def get_worker(scaling_config) -> Worker:
    worker = Worker(
        task_spec=None,
        worker_scaling_config=scaling_config,
    )
    worker.setup()
    return worker


class TestWorker(unittest.TestCase):
    @classmethod
    def setUp(cls) -> None:
        ray.init()

    @classmethod
    def tearDown(cls) -> None:
        ray.shutdown()

    def test_config(self):
        scaling_config = WorkerGroupScalingConfig()
        worker = get_worker(scaling_config)
        result = worker.apply(lambda worker: worker.distributed)
        self.assertEqual(result, scaling_config.num_workers)


if __name__ == "__main__":
    import pytest
    import sys

    sys.exit(pytest.main(["-v", __file__]))
