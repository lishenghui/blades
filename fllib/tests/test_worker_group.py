import time

import ray
import torch

from fllib import session
from fllib.execution.worker import Worker
from fllib.execution.worker_set import WorkerSet


# @pytest.fixture
def ray_start_2_cpus():
    address_info = ray.init(num_cpus=2)
    yield address_info
    # The code after the yield will run as teardown code.
    ray.shutdown()


def test_worker_creation(ray_start_2_cpus):
    assert ray.available_resources()["CPU"] == 2
    wg = WorkerSet(num_workers=2)
    assert len(wg.workers) == 2
    time.sleep(1)
    # Make sure both CPUs are being used by the actors.
    assert "CPU" not in ray.available_resources()
    wg.shutdown()


def test_execute_async(ray_start_2_cpus):
    wg = WorkerSet(
        num_workers=2,
        actor_cls=Worker,
        actor_cls_kwargs={
            "global_model_creator": lambda: torch.nn.Linear(1, 1),
            "global_optimizer_creator": lambda model: torch.optim.SGD(
                model.parameters(), lr=0.1
            ),
        },
        local_worker_cls=Worker,
        local_worker_cls_kwargs={
            "global_model_creator": lambda: torch.nn.Linear(1, 1),
            "global_optimizer_creator": lambda model: torch.optim.SGD(
                model.parameters(), lr=0.1
            ),
        },
    )
    ray.get(wg.execute_async(lambda: session.init_session()))
    futures = wg.execute_async(lambda: session.get_global_model())
    outputs = ray.get(futures)

    state_name = "model"
    wg.foreach_worker(
        lambda w: w.register_state(state_name, torch.rand(2, 2)), with_local=True
    )
    # wg.foreach_worker(lambda _: print("hello"))
    print(outputs)

    wg.sync_state(state_name)

    wg.foreach_worker(lambda w: print(w.get_state(state_name)), with_local=True)
    wg.submit_task(lambda _: print("hello"))


ray_start_2_cpus()
test_execute_async(None)
