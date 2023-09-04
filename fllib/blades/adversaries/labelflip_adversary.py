from fllib.algorithms import Algorithm
from fllib.clients import ClientCallback
from .adversary import Adversary


class LabelFlipAdversary(Adversary):
    def on_algorithm_start(self, algorithm: Algorithm):
        # num_class = self.global_config.dataset_config["custom_dataset_config"][
        num_class = self.global_config.dataset_config["num_classes"]

        class LabelFlipCallback(ClientCallback):
            def on_train_batch_begin(self, data, target):
                return data, num_class - 1 - target

        for client in self.clients:
            client.to_malicious(callbacks_cls=LabelFlipCallback, local_training=True)
