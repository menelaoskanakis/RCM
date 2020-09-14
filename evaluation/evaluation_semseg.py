import numpy as np
import os

from PIL import Image


def eval_semseg(loader, folder, n_classes=20, has_bg=True):

    n_classes = n_classes + int(has_bg)

    # Iterate
    tp = [0] * n_classes
    fp = [0] * n_classes
    fn = [0] * n_classes

    for i, sample in enumerate(loader):
        if i % 500 == 0:
            print('Evaluating: {} of {} objects'.format(i, len(loader)))

        # Load result
        filename = os.path.join(folder, 'semseg', sample['meta']['image'] + '_encoded.png')
        mask = np.array(Image.open(filename)).astype(np.float32)

        gt = sample['labels']['semseg'].numpy()
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

    jac = [0] * n_classes
    for i_part in range(0, n_classes):
        jac[i_part] = float(tp[i_part]) / max(float(tp[i_part] + fp[i_part] + fn[i_part]), 1e-8)

    # Write results
    eval_result = dict()
    eval_result['jaccards_all_categs'] = jac
    eval_result['mIoU'] = np.mean(jac)

    return eval_result
