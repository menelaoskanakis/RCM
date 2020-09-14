from dataloaders.PascalContextMT import PascalContextMT
from dataloaders.NYUDMT import NYUDMT
from torch.utils.data import DataLoader


key2dataset = {
    'PascalContextMT': PascalContextMT,
    'NYUDMT': NYUDMT,
}


def get_dataset(name=None):
    """Get desired dataset

    Args:
        name (string): Name of dataset
    Returns:
        Dataset class
    """
    if name is None:
        raise NotImplementedError("Dataset not defined")

    else:
        if name not in key2dataset:
            raise NotImplementedError("Dataset {} not implemented".format(name))

        return key2dataset[name]


def get_dataloader(dataset_name, tasks_weighting, dataset_args, dataloader_args):
    """Get desired dataloader
    Args:
        dataset_name (str): Name of dataset
        tasks_weighting (dictionary): Dictionary with task and corresponding weight
        dataset_args (dictionary): Arguments to define the dataset
        dataloader_args (dictionary): Arguments to define the dataloader
    Returns:
        Dataloader
    """
    dataset = get_dataset(dataset_name)
    tasks_args = {k: True for k, v in tasks_weighting.items()}
    dataset_args = {**dataset_args, **tasks_args}
    loader = DataLoader(dataset=dataset(**dataset_args),
                        **dataloader_args)
    return loader


def get_test_dataloader(config):
    """
    """
    if config['dataset']['dataset_name'] == 'PascalContextMT':
        dataset = get_dataset('PascalContextMT')
        n_classes = 20
    elif config['dataset']['dataset_name'] == 'NYUDMT':
        dataset = get_dataset('NYUDMT')
        n_classes = 40
    else:
        raise ValueError('Dataset {} is not supported'.format(config['dataset']['dataset_name']))

    tasks_args = {k: True for k, v in config['dataset']['tasks_weighting'].items()}
    dataset_args = {**config['val_dataloader']['dataset_args'], **tasks_args}
    del dataset_args['augmentations']
    return dataset(**dataset_args), tasks_args, n_classes