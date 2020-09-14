import torch
import yaml
import os
import copy

from collections import OrderedDict


def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream)


def get_params(model, lr, num_tasks, task_conditional, tasks):
    if hasattr(model, 'module'):
        model = model.module

    # Set learning rate for common branches according to the number of update per iteration
    if task_conditional:
        multi_backprop_lr = lr / num_tasks
    else:
        multi_backprop_lr = copy.deepcopy(lr)

    # Define appropriate lr for each parameter
    params_list = []
    for name, params in model.named_parameters():
        if params.requires_grad:
            if any(task in name for task in tasks):
                params_list.append({'params': params, 'lr': lr})
            else:
                params_list.append({'params': params, 'lr': multi_backprop_lr})
    return params_list


def activate_gpus(config):
    """Identify which GPU to activate
        Args:
            config: Configuration dictionary with project hyperparameters
        Returns:
            dict: Required information for GPU/CPU training
    """
    if torch.cuda.is_available() and config['gpu_id'] > -1:
        use_gpu = True
        gpu_id = config['gpu_id']
    else:
        use_gpu = False
        gpu_id = []
    device = torch.device("cuda:" + str(gpu_id) if use_gpu else "cpu")
    return {'device': device, 'gpu_id': gpu_id, 'use_gpu': use_gpu}


def mdl_to_device(mdl, gpu_info):
    """Send model to device (GPU/CPU)

    Args:
        mdl (object): Model
        gpu_info(dict): Dictionary with required GPU information
    Returns:
        mdl (object): Model sent to device
    """
    mdl.to(gpu_info['device'])
    return mdl


def get_best_model(dirname, key):
    """Get best model

    Args:
        dirname (str): Directory name
        key(str): Name of the model
    Returns:
        Name of best model
    """
    if os.path.exists(dirname) is False:
        return None
    file_name = key + '_best.pth'
    return os.path.join(dirname, file_name)


def get_last_model(dirname, key):
    """Get best model

    Args:
        dirname (str): Directory name
        key(str): Name of the model
    Returns:
        Name of best model
    """
    if os.path.exists(dirname) is False:
        return None
    file_name = key + '_last.pth'
    return os.path.join(dirname, file_name)


def tensor_to_device(tensor, gpu_info):
    """Send tensor to device (GPU/CPU)

    Args:
        tensor (tensor): Any tensor
        gpu_info(dict): Dictionary with required GPU information
    Returns:
        tensor (tensor): The tensor sent to the device
    """
    return tensor.to(gpu_info['device'])


def dict_to_device(sample, gpu_info):
    """Send dictionary of tensors to device (GPU/CPU)

    Args:
        sample (dict): Dictionary of tensors (image and targets)
        gpu_info(dict): Dictionary with required GPU information
    Returns:
        tensor (tensor): The tensor sent to the device
    """
    sample['image'] = tensor_to_device(sample['image'], gpu_info)

    for key, target in sample['labels'].items():
        sample['labels'][key] = tensor_to_device(sample['labels'][key], gpu_info)
    return sample


def create_dir(directory):
    """Create directory if it does not exist

    Args:
        directory(str): Directory to create
    """
    if not os.path.exists(directory):
        print("Creating directory: {}".format(dir))
        os.makedirs(directory)


def create_results_dir(results_dir, exp_name):
    """Create the required results directory if it does not exist

    Args:
        results_dir(str): Directory to create
        exp_name(str): Name of the experiment to be used in the directory created
    Returns:
        exp_dir (str): Path of experiment directory
        checkpoint_dir (str): Path of checkpoint directory
        img_dir (str): Path of image directory
    """
    create_dir(results_dir)
    exp_dir = os.path.join(results_dir, exp_name)
    create_dir(exp_dir)
    checkpoint_dir = os.path.join(exp_dir, 'checkpoints')
    create_dir(checkpoint_dir)
    img_dir = os.path.join(exp_dir, 'images')
    create_dir(img_dir)
    return exp_dir, checkpoint_dir, img_dir


def create_pred_dir(results_dir, exp_name, config):
    """Create the required results directory if it does not exist

    Args:
        results_dir(str): Directory to create
        exp_name(str): Name of the experiment to be used in the directory created
        config: Configuration dictionary with project hyperparameters
    Returns:
        checkpoint_dir (str): Path of checkpoint directory
        pred_dir (str): Path of experiment directory
    """
    exp_dir = os.path.join(results_dir, exp_name)
    checkpoint_dir = os.path.join(exp_dir, 'checkpoints')
    pred_dir = os.path.join(exp_dir, 'predictions')
    create_dir(pred_dir)

    dataset = config['dataset']
    tasks = [k for k, v in config['dataset']['tasks_weighting'].items()]
    for task in tasks:
        task_dir = os.path.join(pred_dir, task)
        create_dir(task_dir)
    return checkpoint_dir, pred_dir


def write_loss(iteration, writer, model_statistics):
    """Create the required results directory if it does not exist

    Args:
        iteration (int): Current iteration
        writer (object): Writer to log performance
        model_statistics (dict): Statistics to log
    """
    for key, value in model_statistics.items():
        writer.add_scalar(key, value, iteration + 1)


def write_param(iteration, writer, model):
    """Create the required results directory if it does not exist

    Args:
        iteration (int): Current iteration
        writer (object): Writer to log performance
        model: Pytorch model
    """
    for name, param in model.named_parameters():
        writer.add_histogram(name + '_value', param.clone().cpu().data.numpy(), iteration + 1)


def write_grad(iteration, writer, model):
    """Create the required results directory if it does not exist

    Args:
        iteration (int): Current iteration
        writer (object): Writer to log performance
        model: Pytorch model
    """
    for name, param in model.named_parameters():
        if param.grad is not None:
            writer.add_histogram(name + '_grad', param.grad.clone().cpu().data.numpy(), iteration + 1)


def dataset_model_info(name):
    """"""
    # Define the information for the datasets manually
    # Name of task, and information of the tasks including output dim and if normalization is required
    PascalContextMT = OrderedDict([('edge', {'out_dim': 1, 'normalize': False}),
                                   ('human_parts', {'out_dim': 7, 'normalize': False}),
                                   ('semseg', {'out_dim': 21, 'normalize': False}),
                                   ('normals', {'out_dim': 3, 'normalize': True}),
                                   ('sal', {'out_dim': 1, 'normalize': False}),
                                   ])

    NYUDMT = OrderedDict([('edge', {'out_dim': 1, 'normalize': False}),
                          ('semseg', {'out_dim': 41, 'normalize': False}),
                          ('normals', {'out_dim': 3, 'normalize': True}),
                          ('depth', {'out_dim': 1, 'normalize': False}),
                          ])

    Datasets = {'PascalContextMT': PascalContextMT,
                'NYUDMT': NYUDMT}

    if name in Datasets:
        return Datasets[name]
    else:
        raise ValueError('Dataset {} not supported'.format(name))


def save_img(samples, outputs, tasks_weighting, pred_decoder, save_dir):
    img_name = samples['meta']['image'][0]

    # Cut image borders
    orig_img_dim = (samples['meta']['im_size'][0][0].detach().cpu().numpy(),
                    samples['meta']['im_size'][1][0].detach().cpu().numpy())
    current_img_dim = tuple(outputs[list(tasks_weighting.keys())[0]].size()[-2:])

    delta_height = current_img_dim[0] - orig_img_dim[0]
    delta_width = current_img_dim[1] - orig_img_dim[1]

    height_location = [delta_height // 2, (delta_height // 2) + orig_img_dim[0]]
    width_location = [delta_width // 2, (delta_width // 2) + orig_img_dim[1]]
    assert height_location[1] - height_location[0] == orig_img_dim[0]
    assert width_location[1] - width_location[0] == orig_img_dim[1]

    # Save predictions for the different tasks
    for ind, (task, _) in enumerate(tasks_weighting.items()):
        if task in {'sal', 'edge'}:
            pred = torch.sigmoid(outputs[task].squeeze()).detach().cpu().numpy()
            pred_decoder.save_pred(save_dir, img_name, task,
                                   pred[height_location[0]:height_location[1], width_location[0]:width_location[1]])
        elif task in {'human_parts', 'semseg'}:
            pred = torch.argmax(outputs[task].squeeze(), dim=0).detach().cpu().numpy()
            pred_decoder.save_pred(save_dir, img_name, task,
                                   pred[height_location[0]:height_location[1], width_location[0]:width_location[1]])
        elif task in {'normals'}:
            pred = outputs[task].squeeze().detach().cpu().numpy()
            pred_decoder.save_pred(save_dir, img_name, task,
                                   pred[:, height_location[0]:height_location[1], width_location[0]:width_location[1]])
        elif task in {'depth'}:
            pred = outputs[task].squeeze().detach().cpu().numpy()
            pred_decoder.save_pred(save_dir, img_name, task,
                                   pred[height_location[0]:height_location[1], width_location[0]:width_location[1]])
        else:
            raise ValueError('Image saving task {} is not supported'.format(task))


def dic_for_img_vis(samples, outputs, tasks_weighting):
    plot_dictionary = OrderedDict()
    plot_dictionary['image'] = samples['image'].squeeze().cpu().numpy()
    for ind, (task, _) in enumerate(tasks_weighting.items()):
        if task in {'sal', 'edge'}:
            pred = torch.sigmoid(outputs[task].squeeze()).detach().cpu().numpy()
        elif task in {'human_parts', 'semseg'}:
            pred = torch.argmax(outputs[task].squeeze(), dim=0).detach().cpu().numpy()
        elif task in {'normals'}:
            pred = outputs[task].squeeze().detach().cpu().numpy()
        else:
            raise ValueError('Task {} for input decoding is not supported'.format(task))

        label = samples['labels'][task].squeeze().cpu().numpy()
        plot_dictionary[task] = {'pred': pred,
                                 'gt': label,
                                 }
    return plot_dictionary
