import numpy as np
import os

from PIL import Image


def eval_human_parts(loader, folder, n_classes=6, has_bg=True):

    n_classes = n_classes + int(has_bg)

    # Iterate
    tp = [0] * n_classes
    fp = [0] * n_classes
    fn = [0] * n_classes

    counter = 0
    for i, sample in enumerate(loader):
        if i % 500 == 0:
            print('Evaluating: {} of {} objects'.format(i, len(loader)))

        if 'human_parts' not in sample['labels']:
            print('no human parts')
            continue

        # Check for valid pixels
        gt = sample['labels']['human_parts'].numpy()
        uniq = np.unique(gt)
        if len(uniq) == 1 and (uniq[0] == 255 or uniq[0] == 0):
            continue

        # Load result
        filename = os.path.join(folder, 'human_parts', sample['meta']['image'] + '_encoded.png')
        mask = np.array(Image.open(filename)).astype(np.float32)
        counter += 1
        valid = (gt != 255)

        if mask.shape != gt.shape:
            raise ValueError('Prediction and ground truth dimension missmatch')

        # TP, FP, and FN evaluation
        for i_part in range(0, n_classes):
            tmp_gt = (gt == i_part)
            tmp_pred = (mask == i_part)
            tp[i_part] += np.sum(tmp_gt & tmp_pred & valid)
            fp[i_part] += np.sum(~tmp_gt & tmp_pred & valid)
            fn[i_part] += np.sum(tmp_gt & ~tmp_pred & valid)

    print('Successful evaluation for {} images for human parts'.format(counter))

    jac = [0] * n_classes
    for i_part in range(0, n_classes):
        jac[i_part] = float(tp[i_part]) / max(float(tp[i_part] + fp[i_part] + fn[i_part]), 1e-8)

    # Write results
    eval_result = dict()
    eval_result['jaccards_all_categs'] = jac
    eval_result['mIoU'] = np.mean(jac)

    return eval_result
