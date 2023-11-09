import unittest

import ray

from fllib.core.execution.worker_group import WorkerGroup
from fllib.core.execution.worker_group_config import WorkerGroupConfig


class TestWorkerGroup(unittest.TestCase):
    def setUp(self) -> None:
        ray.init()

    def tearDown(self) -> None:
        ray.shutdown()

    def test_execute_with_actor_pool(self):
        """Test synchronous remote calls to only healthy actors."""
        wg_config = WorkerGroupConfig().resources(
            num_remote_workers=2,
            num_cpus_per_worker=1,
            num_gpus_per_worker=0.2,
            local_gpu_idx=0,
        )
        worker_group: WorkerGroup = wg_config.build()

        values = [1, 2, 3, 4]

        results = worker_group.execute_with_actor_pool(
            lambda actor, value: value, values=values
        )
        self.assertEqual(results, values)
        local_results = worker_group.local_execute(
            lambda actor, value: value, values=values
        )
        self.assertEqual(local_results, values)
