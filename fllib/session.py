from typing import Optional

import ray
import torch

from fllib.constants import MAIN_ACTOR


class FLSession:
    def __init__(self) -> None:
        pass

    @property
    def current_actor(self) -> ray.actor.ActorHandle:
        return ray.get_runtime_context().current_actor


_fl_session: Optional[FLSession] = None


def get_current_worker():
    global_worker = ray._private.worker.global_worker
    if MAIN_ACTOR not in global_worker.actors:
        raise ValueError(f"{MAIN_ACTOR} not found in global_worker.actors.")
    return global_worker.actors[MAIN_ACTOR]


def init_session() -> None:
    global _fl_session
    _fl_session = FLSession()


def get_session() -> FLSession:
    global _fl_session
    if _fl_session is None:
        raise Exception("FLSession not initialized")
    return _fl_session


def get_global_model(from_state=False) -> torch.nn.Module:
    main_actor = get_current_worker()
    return main_actor.get_global_model(from_state)


def get_global_optimizer() -> torch.optim.Optimizer:
    main_actor = get_current_worker()
    return main_actor.get_global_optimizer()
