import torch
import numpy as np

from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss, L1Loss
from losses.L1_masked_loss_normals import L1MaskedLossNormals
from losses.L1_masked_loss_depth import L1MaskedLossDepth
from losses.BCE_with_logits_loss_masked import BCEWithLogitsLossMasked
from losses.binary_cross_entropy_with_logits_weighted import BCEWithLogitsLossWeighted

key2loss = {
    "CrossEntropyLoss": CrossEntropyLoss,
    "BCEWithLogitsLoss": BCEWithLogitsLoss,
    "BCEWithLogitsLossMasked": BCEWithLogitsLossMasked,
    "BCEWithLogitsLossWeighted": BCEWithLogitsLossWeighted,
    "L1Loss": L1Loss,
    "L1MaskedLossNormals": L1MaskedLossNormals,
    "L1MaskedLossDepth": L1MaskedLossDepth,
}


def get_loss_function(loss=None):
    """Get loss function

    Args:
        loss (name): Desired loss function to be used
    Returns:
        Loss function
    """
    if loss is None:
        raise NotImplementedError("Loss not defined")

    else:
        if loss not in key2loss:
            raise NotImplementedError("Loss {} not implemented".format(loss))

        return key2loss[loss]


def get_loss_functions(tasks, losses_config):
    loss_cls = {}
    for key in tasks:
        task_config = losses_config[key]
        loss_cl = get_loss_function(task_config['loss_function'])
        if 'parameters' in task_config:
            if 'pos_weight' in task_config['parameters']:
                pos_weight = task_config['parameters']['pos_weight']
                task_config['parameters']['pos_weight'] = torch.from_numpy(np.array(pos_weight))
            loss_cls[key] = loss_cl(**task_config['parameters'])
        else:
            loss_cls[key] = loss_cl()

    return loss_cls
