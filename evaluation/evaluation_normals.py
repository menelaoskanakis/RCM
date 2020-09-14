import numpy as np
import os

from PIL import Image
from collections import OrderedDict


def normalize(arr):
    arr_norm = np.linalg.norm(arr, ord=2, axis=2)[..., np.newaxis] + 1e-12
    return arr / arr_norm


def eval_normals(loader, folder):
    # Iterate
    deg_diff = []
    for i, sample in enumerate(loader):
        if i % 500 == 0:
            print('Evaluating Surface Normals: {} of {} objects'.format(i, len(loader)))

        # Check for valid labels
        label = sample['labels']['normals'].numpy()
        label = np.swapaxes(np.swapaxes(label, 0, 1), 1, 2)

        uniq = np.unique(label)
        if len(uniq) == 1 and uniq[0] == 0:
            continue

        # Load result
        filename = os.path.join(folder, 'normals', sample['meta']['image'] + '.png')
        pred = 2. * np.array(Image.open(filename)).astype(np.float32) / 255.0 - 1.0
        pred = normalize(pred)

        if pred.shape != label.shape:
            raise ValueError('Prediction and ground truth dimension missmatch')

        valid_mask = (np.linalg.norm(label, ord=2, axis=2) != 0)
        pred[np.invert(valid_mask), :] = 0.
        label[np.invert(valid_mask), :] = 0.
        label = normalize(label)

        deg_diff_tmp = np.rad2deg(np.arccos(np.clip(np.sum(pred * label, axis=2), a_min=-1, a_max=1)))
        deg_diff.extend(deg_diff_tmp[valid_mask])

    deg_diff = np.array(deg_diff)
    eval_result = OrderedDict()
    eval_result['mean'] = np.mean(deg_diff)
    eval_result['median'] = np.median(deg_diff)
    eval_result['rmse'] = np.mean(deg_diff ** 2) ** 0.5
    eval_result['11.25'] = np.mean(deg_diff < 11.25) * 100
    eval_result['22.5'] = np.mean(deg_diff < 22.5) * 100
    eval_result['30'] = np.mean(deg_diff < 30) * 100

    return eval_result
