from torch.optim import SGD, Adam, ASGD, Adamax, Adadelta, Adagrad, RMSprop

key2opt = {
    "SGD": SGD,
    "Adam": Adam,
    "ASGD": ASGD,
    "Adamax": Adamax,
    "Adadelta": Adadelta,
    "Adagrad": Adagrad,
    "RMSprop": RMSprop,
}


def get_optimizer(algorithm=None):
    """Get optimizer

    Args:
        algorithm (string): Name of optimizer to be used
    Returns:
        Optimizer
    """
    if algorithm is None:
        print("Optimize using: SGD")
        return SGD
    else:
        if algorithm not in key2opt:
            raise NotImplementedError("Optimizer {} not implemented".format(algorithm))
        print("Optimize using: {}".format(algorithm))

        return key2opt[algorithm]
