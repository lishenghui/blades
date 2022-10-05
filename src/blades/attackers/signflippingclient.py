from blades.core.client import ByzantineClient


class SignflippingClient(ByzantineClient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def on_backward_end(self):
        for name, p in self.global_model.named_parameters():
            p.grad.data = -p.grad.data
