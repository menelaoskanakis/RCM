from modules import deeplab_resnet


def get_module(config, dataset, tasks):
    """Get model

    Args:
        config (dictionary): Name and parameters for the model
        dataset (str): Dataset name
        tasks (list): Name of tasks
    Returns:
        Model
    """
    config['parameters']['dataset'] = dataset
    config['parameters']['tasks'] = tasks

    if config['architecture'] == "resnet18":
        return deeplab_resnet.resnet18(**config['parameters'])
    if config['architecture'] == "resnet34":
        return deeplab_resnet.resnet34(**config['parameters'])
    else:
        raise NotImplementedError("Module {} not defined.".format(config['architecture']))
