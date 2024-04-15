import torch

from fedlib.trainers import Trainer
from fedlib.clients import ClientCallback
from .adversary import Adversary


class LabelFlipAdversary(Adversary):
    def on_trainer_init(self, trainer: Trainer):
        num_class = self._get_num_model_outputs(trainer)

        class LabelFlipCallback(ClientCallback):
            def on_train_batch_begin(self, data, target):
                return data, num_class - 1 - target

        for client in self.clients:
            client.to_malicious(callbacks_cls=LabelFlipCallback, local_training=True)

    def _get_num_model_outputs(self, trainer: Trainer):
        dataset = trainer._dataset
        test_client_0 = dataset.test_client_ids[0]
        client_dataset = dataset.get_client_dataset(test_client_0)
        test_loader = client_dataset.get_test_loader()
        model = trainer.server.get_global_model()
        with torch.no_grad():
            for data, _ in test_loader:
                output = model(data)
                break
        return output.shape[1]
