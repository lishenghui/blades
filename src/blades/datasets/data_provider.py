from . import MNIST, CIFAR10


def get_dataset(name, **kargs):
    dataset = None
    if name == "mnist":
        dataset = MNIST(**kargs)
    elif name == "cifar10":
        dataset = CIFAR10(**kargs)
    return dataset
