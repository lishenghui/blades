import importlib
from attackers.alieclient import AlieClient
from attackers.bitflippingclient import BitflippingClient
from attackers.ipmclient import IpmClient
from attackers.labelflippingclient import LableflippingClient
from attackers.noiseclient import NoiseClient
from simulators.worker import WorkerWithMomentum, RemoteWorker
from tasks.cifar10 import cifar10
from torchvision import datasets


class ClientBuilder(object):
    def __init__(self, options, *args, **kwargs):
        self.options = options

        attacker_path = importlib.import_module('attackers.%sclient' % options.attack)
        attack_client = getattr(attacker_path, options.attack.capitalize() + 'Client')
        
    
    def initialize_workers(self, trainer, worker_rank, model, optimizer, loss_func, device, is_fedavg=False, kwargs=None):
        train_loader = cifar10(
            data_dir=self.options.data_dir,
            data_path=self.options.data_path,
            train=True,
            download=True,
            batch_size=self.options.batch_size,
            # sampler_callback=get_sampler_callback(worker_rank),
            dataset_cls=datasets.CIFAR10,
            drop_last=True,  # Exclude the influence of non-full batch.
            worker_rank=worker_rank,
            **kwargs,
        )
        # NOTE: The first options.num_byzantine nodes are Byzantine
        if worker_rank < self.options.num_byzantine:
            if self.options.attack == "BF":
                return BitflippingClient.options(num_gpus=self.options.gpu_per_actor).remote(
                    data_loader=train_loader,
                    model=model,
                    loss_func=loss_func,
                    device=device,
                    optimizer=optimizer,
                    **kwargs,
                )
            
            if self.options.attack == "LF":
                return LableflippingClient.options(num_gpus=self.options.gpu_per_actor).remote(
                    revertible_label_transformer=lambda target: 9 - target,
                    data_loader=train_loader,
                    model=model,
                    loss_func=loss_func,
                    device=device,
                    optimizer=optimizer,
                    **kwargs,
                )
            
            if self.options.attack == "IPM":
                attacker = IpmClient.remote(
                    epsilon=0.5,
                    is_fedavg=is_fedavg,
                    data_loader=train_loader,
                    model=model,
                    loss_func=loss_func,
                    device=device,
                    optimizer=optimizer,
                    **kwargs,
                )
                attacker.configure.remote(trainer)
                return attacker
            
            if self.options.attack == "IPM_large":
                attacker = IpmClient.options(num_gpus=self.options.gpu_per_actor).remote(
                    epsilon=1000.0,
                    is_fedavg=is_fedavg,
                    data_loader=train_loader,
                    model=model,
                    loss_func=loss_func,
                    device=device,
                    optimizer=optimizer,
                    **kwargs,
                )
                attacker.configure.remote(trainer)
                return attacker
            
            if self.options.attack == "ALIE":
                attacker = AlieClient.options(num_gpus=self.options.gpu_per_actor).remote(
                    n=self.options.num_clients,
                    m=self.options.num_byzantine,
                    is_fedavg=is_fedavg,
                    data_loader=train_loader,
                    model=model,
                    loss_func=loss_func,
                    device=device,
                    optimizer=optimizer,
                    **kwargs,
                )
                attacker.configure.remote(trainer)
                return attacker
            
            if self.options.attack == "noise":
                attacker = NoiseClient.options(num_gpus=self.options.gpu_per_actor).remote(
                    is_fedavg=is_fedavg,
                    data_loader=train_loader,
                    model=model,
                    loss_func=loss_func,
                    device=device,
                    optimizer=optimizer,
                    **kwargs,
                )
                # attacker.configure(trainer)
                return attacker
            
            raise NotImplementedError(f"No such attack {self.options.attack}")
        if self.options.fedavg:
            return RemoteWorker.options(num_gpus=self.options.gpu_per_actor).remote(data_loader=train_loader, model=model,
                                                                               loss_func=loss_func, device=device,
                                                                               optimizer=optimizer, **kwargs, )
        else:
            return WorkerWithMomentum.remote(momentum=self.options.momentum,
                                             data_loader=train_loader,
                                             model=model,
                                             loss_func=loss_func,
                                             device=device,
                                             optimizer=optimizer,
                                             **kwargs, )