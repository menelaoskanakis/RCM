import argparse
import torch.backends.cudnn as cudnn
import os
import numpy as np

from trainer import Trainer
from dataloaders import get_dataloader, get_test_dataloader
from util import get_config, activate_gpus, mdl_to_device, create_pred_dir
from dataloaders import get_dataset
from evaluation import *
from utils import get_timer

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, help='Path to the config file used for training.')
parser.add_argument('--gpu_id', type=int, default='-1', help='gpu id: e.g. 0 1. use -1 for CPU')
parser.add_argument('--log_performance', type=str, help='Path to log performance.')
opts = parser.parse_args()

cudnn.benchmark = True
# Load experiment setting
config = get_config(opts.config)
logger_config = config['logger']
config['gpu_id'] = opts.gpu_id
gpu_info = activate_gpus(config)

# Setup experiment model
trainer = Trainer(config)
trainer = mdl_to_device(trainer, gpu_info)

# Get data loaders for prediction and final evaluation
val_loader = get_dataloader(**config['dataset'], **config['val_dataloader'])
test_loader, tasks_args, semseg_classes = get_test_dataloader(config)

# Setup output folders
exp_name = os.path.splitext(os.path.basename(opts.config))[0]
results_dir = logger_config['results_dir']
checkpoint_dir, pred_dir = create_pred_dir(results_dir, exp_name, config)

# Save all predictions
with get_timer("Elapsed time: %f"):
    # Evaluation step
    trainer.load_last_model(checkpoint_dir)
    trainer.evaluate_model(val_loader, gpu_info, pred_dir)

results = exp_name
if 'semseg' in tasks_args:
    print('Semantic Segmentation Results:')

    results = '{0:s}, {1:s}'.format(results, 'semseg')
    semseg_results = eval_semseg(test_loader, pred_dir, semseg_classes)
    class_IoU = semseg_results['jaccards_all_categs']
    mIoU = semseg_results['mIoU']

    # Print Results
    print('Semantic Segmentation mIoU: {0:.4f}'.format(100 * mIoU))
    results = '{0:s}, {1:s}, {2:.4f}'.format(results, 'mIoU', 100 * mIoU)
    for i in range(len(class_IoU)):
        print('Category {0:s} {1:.4f}'.format(str(i), 100 * class_IoU[i]))
        results = '{0:s}, {1:s}, {2:.4f}'.format(results, 'category_' +str(i), 100 * class_IoU[i])

if 'edge' in tasks_args:
    results = '{0:s}, {1:s}, {2:s}'.format(results, 'edge', 'done manually')
    print('Evaluation for edges is done manually using the SEISM repo and the evaluation script in '
          './evaluation/evaluation_edge.py')

if 'human_parts' in tasks_args:
    print('Human Parts Results:')

    results = '{0:s}, {1:s}'.format(results, 'human_parts')
    human_parts_results = eval_human_parts(test_loader, pred_dir)
    class_IoU = human_parts_results['jaccards_all_categs']
    mIoU = human_parts_results['mIoU']

    # Print Results
    print('Human Parts mIoU: {0:.4f}'.format(100 * mIoU))
    results = '{0:s}, {1:s}, {2:.4f}'.format(results, 'mIoU', 100 * mIoU)
    for i in range(len(class_IoU)):
        print('Category {0:s} {1:.4f}'.format(str(i), 100 * class_IoU[i]))
        results = '{0:s}, {1:s}, {2:.4f}'.format(results, 'category_' +str(i), 100 * class_IoU[i])

if 'normals' in tasks_args:
    print('Normals Results:')

    results = '{0:s}, {1:s}'.format(results, 'normals')
    normals_results = eval_normals(test_loader, pred_dir)

    # Print Results
    for key, value in normals_results.items():
        print('Result for {0:s}: {1:.4f}'.format(key, value))
        results = '{0:s}, {1:s}, {2:.4f}'.format(results, key, value)

if 'sal' in tasks_args:
    print('Saliency Results:')

    results = '{0:s}, {1:s}'.format(results, 'sal')
    sal_results = eval_sal(test_loader, pred_dir, mask_thres=np.linspace(0.2, 0.9, 15))
    mIoU = sal_results['mIoU']
    maxF = sal_results['maxF']

    # Print Results
    print('Results for Saliency Estimation:')
    print('mIoU: {0:.4f}'.format(mIoU))
    results = '{0:s}, {1:s}, {2:.4f}'.format(results, 'mIoU', 100 * mIoU)
    print('maxF: {0:.4f}'.format(maxF))
    results = '{0:s}, {1:s}, {2:.4f}'.format(results, 'maxF', 100 * maxF)

if 'depth' in tasks_args:
    print('Depth Results:')

    results = '{0:s}, {1:s}'.format(results, 'depth')
    depth_results = eval_depth(test_loader, pred_dir)
    rmse = depth_results['rmse']
    log_rmse = depth_results['log_rmse']

    # Print Results
    print('rmse: {0:.4f}'.format(rmse))
    results = '{0:s}, {1:s}, {2:.4f}'.format(results, 'rmse', rmse)
    print('log_rmse: {0:.4f}'.format(log_rmse))
    results = '{0:s}, {1:s}, {2:.4f}'.format(results, 'log_rmse', log_rmse)

results = '{0:s}\n'.format(results)
print(results)
if os.path.exists(opts.log_performance):
    mode = 'a'
else:
    mode = 'w'
text_file = open(opts.log_performance, mode)
text_file.write(results)
text_file.close()
