import torch

from fllib.algorithms import Algorithm
from fllib.clients import ClientCallback
from .adversary import Adversary


class LabelFlipAdversary(Adversary):
    def on_algorithm_start(self, algorithm: Algorithm):
        num_class = self._get_num_model_outputs(algorithm)

        class LabelFlipCallback(ClientCallback):
            def on_train_batch_begin(self, data, target):
                return data, num_class - 1 - target

        for client in self.clients:
            client.to_malicious(callbacks_cls=LabelFlipCallback, local_training=True)

    def _get_num_model_outputs(self, algorithm: Algorithm):
        dataset = algorithm._dataset
        test_client_0 = dataset.test_client_ids[0]
        client_dataset = dataset.get_client_dataset(test_client_0)
        test_loader = client_dataset.get_test_loader()
        model = algorithm.server.get_global_model()
        with torch.no_grad():
            for data, _ in test_loader:
                output = model(data)
                break
        return output.shape[1]
