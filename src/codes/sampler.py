import math
import torch
import torch.distributed as dist
from torch.utils.data.sampler import Sampler


class DistributedSampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.

    .. note::
        Dataset is assumed to be of constant size.

    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (optional): Number of processes participating in distributed training.
        rank (optional): Rank of the current process within num_replicas.
        shuffle (optional): If true (default), sampler will shuffle the indices
    """
    
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
    
    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        if self.shuffle:
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))
        
        # add extra samples to make it evenly divisible
        indices += indices[: (self.total_size - len(indices))]
        assert len(indices) == self.total_size
        
        # subsample
        indices = indices[self.rank: self.total_size: self.num_replicas]
        assert len(indices) == self.num_samples
        
        return iter(indices)
    
    def __len__(self):
        return self.num_samples
    
    def set_epoch(self, epoch):
        self.epoch = epoch
    
    def __str__(self):
        return "DistributedSampler(num_replicas={num_replicas},rank={rank},shuffle={shuffle})".format(
            num_replicas=self.num_replicas, rank=self.rank, shuffle=self.shuffle
        )


class DecentralizedNonIIDSampler(DistributedSampler):
    def __iter__(self):
        nlabels = len(self.dataset.classes)
        
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(0)
        
        indices = []
        for i in range(nlabels):
            indices_i = torch.nonzero(self.dataset.targets == i)
            
            indices_i = indices_i.flatten().tolist()
            indices += indices_i
        
        # add extra samples to make it evenly divisible
        indices += indices[: (self.total_size - len(indices))]
        assert len(indices) == self.total_size
        
        # subsample
        indices = indices[
                  self.rank * self.num_samples: (self.rank + 1) * self.num_samples
                  ]
        assert len(indices) == self.num_samples
        
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            idx_idx = torch.randperm(len(indices), generator=g).tolist()
            indices = [indices[i] for i in idx_idx]
        
        return iter(indices)
    
    def __str__(self):
        return "DecentralizedNonIIDSampler(num_replicas={num_replicas},rank={rank},shuffle={shuffle})".format(
            num_replicas=self.num_replicas, rank=self.rank, shuffle=self.shuffle
        )
