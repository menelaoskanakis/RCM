from schedulers.schedulers import ConstantLR, PolynomialLR


key2scheduler = {
    "ConstantLR": ConstantLR,
    "PolynomialLR": PolynomialLR,
}


def get_scheduler(lr_policy=None):
    """Get scheduler

    Args:
        lr_policy (string): name of learning rate policy
    Returns:
        Get learning rate policy
    """
    if lr_policy is None:
        print("LR scheduler: {}".format('ConstantLR'))
        return ConstantLR

    else:
        if lr_policy not in key2scheduler:
            raise NotImplementedError("learning rate policy {} not implemented".format(lr_policy))
        print("LR scheduler: {}".format(lr_policy))
        return key2scheduler[lr_policy]
