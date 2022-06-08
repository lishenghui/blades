import torch

from blades.client import ByzantineClient


class NoiseClient(ByzantineClient):
    def __init__(self, noise=0.1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__noise = noise
    
    def omniscient_callback(self, simulator):
        self.state['saved_update'] = torch.normal(self.__noise,
                                                  self.__noise,
                                                  size=super().get_update().shape
                                                  ).to('cpu')
