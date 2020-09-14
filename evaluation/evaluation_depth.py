import os.path
import numpy as np
import scipy.io as sio


def eval_depth(loader, folder):

    rmses = []
    log_rmses = []
    for i, sample in enumerate(loader):

        if i % 500 == 0:
            print('Evaluating depth: {} of {} objects'.format(i, len(loader)))

        # Load result
        filename = os.path.join(folder, 'depth', sample['meta']['image'] + '.mat')

        pred = sio.loadmat(filename)['depth'].astype(np.float32)

        label = sample['labels']['depth'].numpy()

        if pred.shape != label.shape:
            raise ValueError('Prediction and ground truth dimension missmatch')

        label[label == 0] = 1e-9
        pred[pred <= 0] = 1e-9

        valid_mask = (label != 0)
        pred[np.invert(valid_mask)] = 0.
        label[np.invert(valid_mask)] = 0.
        n_valid = np.sum(valid_mask)

        log_rmse_tmp = (np.log(label) - np.log(pred)) ** 2
        log_rmse_tmp = np.sqrt(np.sum(log_rmse_tmp) / n_valid)
        log_rmses.extend([log_rmse_tmp])

        rmse_tmp = (label - pred) ** 2
        rmse_tmp = np.sqrt(np.sum(rmse_tmp) / n_valid)
        rmses.extend([rmse_tmp])

    rmses = np.array(rmses)
    log_rmses = np.array(log_rmses)

    eval_result = dict()
    eval_result['rmse'] = np.mean(rmses)
    eval_result['log_rmse'] = np.median(log_rmses)

    return eval_result
