# Represents a BaseEnv, MultiAgentEnv, ExternalEnv, ExternalMultiAgentEnv,
# VectorEnv, gym.Env, or ActorHandle.
from typing import Callable

DatasetCreator = Callable
ModelCreator = Callable
OptCreator = Callable

# Type of dict returned by get_weights() representing model weights.
ModelWeights = dict

# Represents the result dict returned by Algorithm.train().
ResultDict = dict

# Constant that's True when type checking, but False here.
TYPE_CHECKING = False

# An algorithm config dict that only has overrides. It needs to be combined with
# the default algorithm config to be used.
PartialAlgorithmConfigDict = PartialTrainerConfigDict = dict

# Represents a fully filled out config of a Algorithm class.
# Note: Policy config dicts are usually the same as AlgorithmConfigDict, but
# parts of it may sometimes be altered in e.g. a multi-agent setup,
# where we have >1 Policies in the same Algorithm.
AlgorithmConfigDict = TrainerConfigDict = dict


# Represents the model config sub-dict of the algo config that is passed to
# the model catalog.
ModelConfigDict = dict

DatasetConfigDict = dict

# Represents a generic identifier for a client (e.g., "client1").
ClientID = str


class _NotProvided:
    """Singleton class to provide a "not provided" value for AlgorithmConfig
    signatures.

    Using the only instance of this class indicates that the user does NOT wish to
    change the value of some property.

    Examples:
        >>> from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
        >>> config = AlgorithmConfig()
        >>> # Print out the default learning rate.
        >>> print(config.lr)
        ... 0.001
        >>> # Print out the default `preprocessor_pref`.
        >>> print(config.preprocessor_pref)
        ... "deepmind"
        >>> # Will only set the `preprocessor_pref` property (to None) and leave
        >>> # all other properties at their default values.
        >>> config.training(preprocessor_pref=None)
        >>> config.preprocessor_pref is None
        ... True
        >>> # Still the same value (didn't touch it in the call to `.training()`.
        >>> print(config.lr)
        ... 0.001
    """

    class __NotProvided:
        pass

    instance = None

    def __init__(self):
        if _NotProvided.instance is None:
            _NotProvided.instance = _NotProvided.__NotProvided()


# Use this object as default values in all method signatures of
# AlgorithmConfig, indicating that the respective property should NOT be touched
# in the call.
NotProvided = _NotProvided()
