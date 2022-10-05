from blades.datasets import MNIST, CIFAR10, CIFAR100


def test_dataset():
    mnist = MNIST()
    # local_data = mnist.get_train_loader(u_id="0")
    assert mnist.num_classes == 10

    cifar10 = CIFAR10()
    assert cifar10.num_classes == 10

    cifar10 = CIFAR100()
    assert cifar10.num_classes == 100
