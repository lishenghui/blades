import ray

from fllib.constants import MAIN_ACTOR


def get_session(client_id):
    global_worker = ray._private.worker.global_worker
    if MAIN_ACTOR not in global_worker.actors:
        raise ValueError(f"{MAIN_ACTOR} not found in global_worker.actors.")
    main_actor = global_worker.actors[MAIN_ACTOR]
    main_actor.switch_client(client_id)
    return main_actor
