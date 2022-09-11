import inspect
import os
import sys

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from blades.client import ByzantineClient


class LabelflippingClient(ByzantineClient):
    def __init__(self, num_classes=10, *args, **kwargs):
        """
        Args:
            revertible_label_transformer (callable):
                E.g. lambda label: 9 - label
        """
        super().__init__(*args, **kwargs)
        self.num_classes = num_classes
    
    def on_train_batch_begin(self, data, target, logs=None):
        return data, self.num_classes - 1 - target
    
    def __str__(self) -> str:
        return "LableFlippingWorker"
