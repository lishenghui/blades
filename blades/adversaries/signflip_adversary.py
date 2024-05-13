from fedlib.trainers import Trainer
from fedlib.clients import ClientCallback
from .adversary import Adversary


class SignFlipAdversary(Adversary):
    def on_trainer_init(self, trainer: Trainer):
        class SignFlipCallback(ClientCallback):
            def on_backward_end(self, task):
                model = task.model
                for _, para in model.named_parameters():
                    para.grad.data = -para.grad.data

        for client in self.clients:
            client.to_malicious(callbacks_cls=SignFlipCallback, local_training=True)
