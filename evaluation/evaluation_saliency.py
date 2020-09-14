import numpy as np
import os

from PIL import Image


def jaccard(gt, pred, void_pixels=None):

    assert(gt.shape == pred.shape)

    if void_pixels is None:
        void_pixels = np.zeros_like(gt)
    assert(void_pixels.shape == gt.shape)

    gt = gt.astype(np.bool)
    pred = pred.astype(np.bool)
    void_pixels = void_pixels.astype(np.bool)
    if np.isclose(np.sum(gt & np.logical_not(void_pixels)), 0) and np.isclose(np.sum(pred & np.logical_not(void_pixels)), 0):
        return 1
    else:
        return np.sum(((gt & pred) & np.logical_not(void_pixels))) / \
               np.sum(((gt | pred) & np.logical_not(void_pixels)), dtype=np.float32)


def precision_recall(gt, pred, void_pixels=None):

    if void_pixels is None:
        void_pixels = np.zeros_like(gt)

    gt = gt.astype(np.bool)
    pred = pred.astype(np.bool)
    void_pixels = void_pixels.astype(np.bool)

    tp = ((pred & gt) & ~void_pixels).sum()
    fn = ((~pred & gt) & ~void_pixels).sum()

    fp = ((pred & ~gt) & ~void_pixels).sum()

    prec = tp / (tp + fp + 1e-12)
    rec = tp / (tp + fn + 1e-12)

    return prec, rec


def eval_sal(loader, folder, mask_thres=None):
    if mask_thres is None:
        mask_thres = [0.5]

    eval_result = dict()
    eval_result['all_jaccards'] = np.zeros((len(loader), len(mask_thres)))
    eval_result['prec'] = np.zeros((len(loader), len(mask_thres)))
    eval_result['rec'] = np.zeros((len(loader), len(mask_thres)))

    for i, sample in enumerate(loader):
        if i % 500 == 0:
            print('Evaluating: {} of {} objects'.format(i, len(loader)))

        # Load result
        filename = os.path.join(folder, 'sal', sample['meta']['image'] + '.png')
        mask = np.array(Image.open(filename)).astype(np.float32) / 255.
        gt = sample['labels']['sal'].numpy()

        if mask.shape != gt.shape:
            raise ValueError('Prediction and ground truth dimension missmatch')

        for j, thres in enumerate(mask_thres):
            gt = (gt > thres).astype(np.float32)
            mask_eval = (mask > thres).astype(np.float32)
            eval_result['all_jaccards'][i, j] = jaccard(gt, mask_eval)
            eval_result['prec'][i, j], eval_result['rec'][i, j] = precision_recall(gt, mask_eval)

    # Average for each thresholds
    eval_result['mIoUs'] = np.mean(eval_result['all_jaccards'], 0)
    eval_result['mPrec'] = np.mean(eval_result['prec'], 0)
    eval_result['mRec'] = np.mean(eval_result['rec'], 0)
    eval_result['F'] = 2 * eval_result['mPrec'] * eval_result['mRec'] / \
                       (eval_result['mPrec'] + eval_result['mRec'] + 1e-12)

    # Maximum of averages (maxF, maxmIoU)
    eval_result['mIoU'] = np.max(eval_result['mIoUs'])
    eval_result['maxF'] = np.max(eval_result['F'])

    return eval_result
