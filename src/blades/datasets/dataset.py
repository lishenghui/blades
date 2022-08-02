from warnings import warn


# class BaseDataset(ABC):
#     def __init__(
#             self,
#             data_root: str = './data',
#             train_bs: Optional[int] = 32,
#             iid: Optional[bool] = True,
#             alpha: Optional[float] = 0.1,
#             num_clients: Optional[int] = 20
#     ):
#         self.train_bs = train_bs
#         self._data_path = os.path.join(data_root, self.__class__.__name__ + '.obj')
#         if not os.path.exists(self._data_path):
#             self.generate_datasets(data_root, iid, alpha, num_clients)
#
#     @abstractmethod
#     def generate_datasets(self, path='./data', iid=True, alpha=0.1, num_clients=20):
#         pass
#
#     @staticmethod
#     def _preprocess_train_data(
#             data,
#             labels,
#             batch_size,
#             seed=0
#     ) -> (torch.Tensor, torch.LongTensor):
#         i = 0
#         # The following line is needed for reproducing the randomness of transforms.
#         set_random_seed(seed)
#
#         idx = np.random.permutation(len(labels))
#         data, labels = data[idx], labels[idx]
#
#         while True:
#             if i * batch_size >= len(labels):
#                 i = 0
#                 idx = np.random.permutation(len(labels))
#                 data, labels = data[idx], labels[idx]
#
#                 continue
#             else:
#                 X = data[i * batch_size:(i + 1) * batch_size, :]
#                 y = labels[i * batch_size:(i + 1) * batch_size]
#                 i += 1
#                 X = torch.Tensor(X)
#                 yield X, torch.LongTensor(y)
#
#     @staticmethod
#     def _preprocess_test_data(
#             data,
#             labels,
#     ) -> CustomTensorDataset:
#         tensor_x = torch.Tensor(data)  # transform to torch tensor
#         tensor_y = torch.LongTensor(labels)
#         return CustomTensorDataset(tensor_x, tensor_y)
#
#     # generate two lists of dataloaders for train
#     def get_dls(self):
#         assert os.path.isfile(self._data_path)
#         with open(self._data_path, 'rb') as f:
#             (train_clients, train_data, test_clients, test_data) = [pickle.load(f) for _ in range(4)]
#
#         assert sorted(train_clients) == sorted(test_clients)
#
#         train_dls = []
#         test_dls = []
#         for idx, u_id in enumerate(train_clients):
#             train_dls.append(self._preprocess_train_data(data=np.array(train_data[u_id]['x']),
#                                                          labels=np.array(train_data[u_id]['y']),
#                                                          batch_size=self.train_bs,
#                                                          ))
#             test_dls.append(self._preprocess_test_data(data=np.array(test_data[u_id]['x']),
#                                                        labels=np.array(test_data[u_id]['y']),
#                                                        ))
#         return train_dls, test_dls
#
    
class FLDataset(object):
    """ Federated Larning dataset

    Args:
        train_dataloaders (list): dataloaders of training data for input
        test_dataloaders (list, optional): dataloaders of test data for input. Two lists of dataloders should be
            in corresponding order to the client they belong to.
    """
    
    def __init__(
            self,
            train_dataloaders: list,
            test_dataloaders: list = None
            # Input: train and test dataloaders for each client in corresponding order
    ) -> None:
        if not test_dataloaders:
            warn("No test data is given. Model evaluation will be based on train data. ")
            test_dataloaders = train_dataloaders
        if len(train_dataloaders) != len(test_dataloaders):
            raise Exception("Invalid Input: Numbers of train dataloaders and test dataloaders should be equal. ")
        self._train_dls = {}
        self._test_dls = {}
        for idx, (traindl, testdl) in enumerate(zip(train_dataloaders, test_dataloaders)):
            self._train_dls[idx] = traindl
            self._test_dls[idx] = testdl
        self._clients = list(range(len(self._train_dls)))
    
    def get_clients(self):
        return self._clients
    
    def get_train_data(self, u_id, num_batches):
        data = [next(self._train_dls[u_id]) for _ in range(num_batches)]
        return data
    
    def get_all_test_data(self, u_id):
        return self._test_dls[u_id]
