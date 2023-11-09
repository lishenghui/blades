import warnings
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from fllib.algorithms import Algorithm


class AlgorithmCallback:
    """Abstract base class for RLlib callbacks (similar to Keras callbacks).

    These callbacks can be used for custom metrics and custom postprocessing.

    By default, all of these callbacks are no-ops. To configure custom training
    callbacks, subclass DefaultCallbacks and then set
    {"callbacks": YourCallbacksClass} in the algo config.
    """

    def __init__(self) -> None:
        self._algorithm = None

    def setup(
        self,
        algorithm: "Algorithm",
        **info,
    ):
        self._algorithm = algorithm

    def on_train_round_begin(self) -> None:
        """Called at the beginning of each local training round in
        `train_global_model` methods.

        Subclasses should override for any actions to run.
        Returns:
        """

    def on_train_round_end(self):
        """A callback method called after local training.

        It is typically used to modify updates (i.e,. pseudo-gradient).
        """


class AlgorithmCallbackList:
    def __init__(self, callbacks: List[AlgorithmCallback]):
        self._callbacks = callbacks
        self._algorithm = None

    def setup(self, algorithm, **info):
        self._algorithm = algorithm
        for callback in self._callbacks:
            try:
                callback.setup(algorithm, **info)
            except TypeError as e:
                if "argument" in str(e):
                    warnings.warn(
                        "Please update `setup` method in callback "
                        f"`{callback.__class__}` to match the method signature"
                        " in `ray.tune.callback.Callback`.",
                        FutureWarning,
                    )
                    callback.setup()
                else:
                    raise e

    def append(self, callback: AlgorithmCallback):
        callback.setup(self._algorithm)
        self._callbacks.append(callback)

    def on_train_round_begin(self) -> None:
        for callback in self._callbacks:
            callback.on_train_round_begin()

    def on_train_round_end(self):
        for callback in self._callbacks:
            callback.on_train_round_end()
