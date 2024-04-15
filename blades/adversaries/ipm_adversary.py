from fedlib.trainers import Trainer
from fedlib.constants import CLIENT_UPDATE, CLIENT_ID
from .adversary import Adversary


class IPMAdversary(Adversary):
    def __init__(self, scale: float = 1.0):
        super().__init__()

        self._scale = scale

    def on_local_round_end(self, algorithm: Trainer):
        benign_updates = self.get_benign_updates(algorithm)
        mean = benign_updates.mean(dim=0)

        update = -self._scale * mean
        for result in algorithm.local_results:
            client = algorithm.client_manager.get_client_by_id(result[CLIENT_ID])
            if client.is_malicious:
                result[CLIENT_UPDATE] = update
