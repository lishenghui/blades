from warnings import warn


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
