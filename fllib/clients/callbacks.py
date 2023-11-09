import warnings
from typing import List


class ClientCallback:
    """Abstract base class for RLlib callbacks (similar to Keras callbacks).

    These callbacks can be used for custom metrics and custom postprocessing.

    By default, all of these callbacks are no-ops. To configure custom training
    callbacks, subclass DefaultCallbacks and then set
    {"callbacks": YourCallbacksClass} in the algo config.
    """

    def __init__(self) -> None:
        self._client = None

    def setup(
        self,
        client,
        **info,
    ):
        self._client = client

    def on_train_round_begin(self) -> None:
        """Called at the beginning of each local training round in
        `train_global_model` methods.

        Subclasses should override for any actions to run.
        Returns:
        """

    def on_train_batch_begin(self, data, target):
        """Called at the beginning of a training batch in `train_global_model`
        methods.

        Subclasses should override for any actions to run.
        Args:
            data: input of the batch data.
            target: target of the batch data.
        """
        return data, target

    def on_backward_end(self, task):
        """A callback method called after backward and before parameter update.

        It is typically used to modify gradients.
        """
        pass

    def on_train_round_end(self):
        """A callback method called after local training.

        It is typically used to modify updates (i.e,. pseudo-gradient).
        """
        pass


class ClientCallbackList:
    def __init__(self, callbacks: List[ClientCallback]):
        self._callbacks = callbacks
        self._client = None

    def setup(self, client, **info):
        self._client = client
        for callback in self._callbacks:
            try:
                callback.setup(client, **info)
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

    def append(self, callback: ClientCallback):
        callback.setup(self._client)
        self._callbacks.append(callback)

    def on_train_round_begin(self) -> None:
        for callback in self._callbacks:
            callback.on_train_round_begin()

    def on_train_batch_begin(self, data, target):
        for callback in self._callbacks:
            data, target = callback.on_train_batch_begin(data, target)
        return data, target

    def on_backward_end(self, task):
        for callback in self._callbacks:
            callback.on_backward_end(task)

    def on_train_round_end(self):
        for callback in self._callbacks:
            callback.on_train_round_end()
