def load(dataset, model_name, model_file, data_parallel=False):
    if dataset == 'cifar10':
        net = others.landscape.cifar10.model_loader.load(model_name, model_file, data_parallel)
    return net