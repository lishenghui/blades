# import unittest
# import torch
# from fllib.models.catalog import ModelCatalog


# class TestModels(unittest.TestCase):
#     def test_custom_model(self):
#         def BinaryClassificationModel():
#             return torch.nn.Sequential(torch.nn.Linear(2, 1), torch.nn.Sigmoid())

#         ModelCatalog.register_custom_model(
#             "BinaryClassificationModel", BinaryClassificationModel
#         )
#         model = ModelCatalog.get_model("BinaryClassificationModel")
#         self.assertIsInstance(model, torch.nn.Sequential)


# if __name__ == "__main__":
#     unittest.main()

import unittest

import torch

from fllib.models.catalog import ModelCatalog


class TestModels(unittest.TestCase):
    def test_custom_model_registration(self):
        # Define a custom binary classification model
        def simple_model():
            return torch.nn.Sequential(torch.nn.Linear(2, 1), torch.nn.Sigmoid())

        # Register the custom model with the model catalog
        ModelCatalog.register_custom_model("SimpleModel", simple_model)

        # Retrieve the custom model from the model catalog
        model = ModelCatalog.get_model({"custom_model": "SimpleModel"})

        # Ensure that the retrieved model is an instance of torch.nn.Sequential
        self.assertIsInstance(model, torch.nn.Sequential)


if __name__ == "__main__":
    unittest.main()
