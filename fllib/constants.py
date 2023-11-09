import os
import sys

from ray.tune.registry import RLLIB_MODEL, TRAINABLE_CLASS, RLLIB_ACTION_DIST


def env_integer(key, default):
    if key in os.environ:
        val = os.environ[key]
        if val == "inf":
            return sys.maxsize
        else:
            return int(val)
    return default


# Counters for training steps.
NUM_GLOBAL_STEPS = "num_global_steps"

# How long to wait for a communicator to read from the message queue.
COMMUNICATOR_INTERVAL_S = env_integer("COMMUNICATOR_INTERVAL_S", 0.01)

PROCESS_TYPE_COMMUNICATOR = "communicator"

MINIMUM_GPU_FRACTION = 0.0001

SERVER_RANK = 0

DEFAULT_MASTER_PORT = "22225"

DEFAULT_DATA_ROOT = "~/fldata"

MAIN_ACTOR = "main_actor"

GLOBAL_MODEL = "global_model"
CLIENT_UPDATE = "client_update"

# Map fllib categories to rllib categories supported by Tune.
# This is a temporary solution as we cannot add new categories to Tune so far.
# https://github.com/ray-project/ray/blob/5568b6142f0e6eb60165265e5415fda6342eb15f/python/ray/tune/registry.py#L26
FLLIB_MODEL = RLLIB_MODEL
FLLIB_ALGORITHM = TRAINABLE_CLASS
FLLIB_DATASET = RLLIB_ACTION_DIST
