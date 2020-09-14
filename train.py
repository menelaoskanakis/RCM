import argparse
import torch.backends.cudnn as cudnn
import sys
import os
import shutil

from trainer import Trainer
from torch.utils.tensorboard import SummaryWriter
from dataloaders import get_dataloader
from utils import get_timer
from util import get_config, write_loss, activate_gpus, mdl_to_device, create_results_dir, dict_to_device, write_grad, \
    write_param


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, help='Path to the config file.')
parser.add_argument('--resume', type=bool, default=False, help='Resume training.')
parser.add_argument('--gpu_id', type=int, default='-1', help='gpu id: e.g. 0 1. use -1 for CPU')
opts = parser.parse_args()

cudnn.benchmark = True

# Load experiment settings
config = get_config(opts.config)
training_strategy_config = config['training_strategy']
max_iter = training_strategy_config['max_iter']
logger_config = config['logger']
config['gpu_id'] = opts.gpu_id
gpu_info = activate_gpus(config)

# Get data loaders
train_loader = get_dataloader(**config['dataset'], **config['train_dataloader'])
val_loader = get_dataloader(**config['dataset'], **config['val_dataloader'])
img_visualization_loader = get_dataloader(**config['dataset'], **config['sample_imgs_dataloader'])

# Setup experiment model
trainer = Trainer(config)
trainer = mdl_to_device(trainer, gpu_info)

# Setup logger and output folders
exp_name = os.path.splitext(os.path.basename(opts.config))[0]
results_dir = logger_config['results_dir']
exp_dir, checkpoint_dir, img_dir = create_results_dir(results_dir, exp_name)
train_summary = SummaryWriter(os.path.join(exp_dir, 'logs/train_weight'))
val_summary = SummaryWriter(os.path.join(exp_dir, 'logs/val'))
img_summary = SummaryWriter(img_dir)
shutil.copy(opts.config, os.path.join(exp_dir, 'config.yaml'))

# Commence with new training or resume from checkpoint
iteration = trainer.resume(checkpoint_dir, val_loader, gpu_info) if opts.resume else 0
with get_timer("Elapsed time: %f"):
    while True:
        for it, (samples) in enumerate(train_loader):
            samples = dict_to_device(samples, gpu_info)

            # Weights training step
            trainer.update_model(samples)

            # Training logging step
            if (iteration + 1) % logger_config['train_log_iter'] == 0 or (iteration + 1) >= max_iter:
                print("Iteration - Weights update: %08d/%08d" % (iteration + 1, max_iter))
                model_statistics = trainer.get_train_loss()
                write_loss(iteration, train_summary, model_statistics)
                write_param(iteration, train_summary, trainer)
                write_grad(iteration, train_summary, trainer)

            # Evaluation and logging step
            if (iteration + 1) % logger_config['val_log_iter'] == 0 or (iteration + 1) >= max_iter:
                trainer.evaluate_model(val_loader, gpu_info)
                model_statistics = trainer.get_val_loss()
                write_loss(iteration, val_summary, model_statistics)
                trainer.save(checkpoint_dir, iteration, model_statistics)

            # Visual inspection logging. Only supports Pascal Context
            if (iteration + 1) % logger_config['image_log_iter'] == 0 or (iteration + 1) >= max_iter:
                trainer.img_visualization(img_visualization_loader, gpu_info, it, img_summary)

            iteration += 1
            if iteration >= max_iter:

                train_summary.close()
                val_summary.close()
                img_summary.close()

                cmd = 'rm {}/model_0*'.format(checkpoint_dir)
                os.system(cmd)
                cmd = 'rm {}/optimizer*'.format(checkpoint_dir)
                os.system(cmd)

                sys.exit('Training finished')
