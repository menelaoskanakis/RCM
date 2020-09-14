import torch
import torch.nn as nn
import os
import shutil

from util import get_best_model, get_last_model, dict_to_device
from utils import get_running_meter, get_img_logging, get_pred_decoder
from optimizers import get_optimizer
from schedulers import get_scheduler
from losses import get_loss_functions
from modules import get_module
from util import get_params, save_img, dic_for_img_vis


class Trainer(nn.Module):
    def __init__(self, config):
        """Initialize Trainer

        Args:
            config (dict): Configuration dictionary
        """
        super(Trainer, self).__init__()

        # Define multi-task setting
        dataset = config['dataset']
        dataset_name = dataset['dataset_name']
        self.tasks_weighting = dataset['tasks_weighting']
        self.tasks = [k for k, v in self.tasks_weighting.items()]

        # Setup network
        model_config = config['model']
        self.model = get_module(model_config, dataset_name, self.tasks)
        print('Model constructed for {}'.format(' '.join(self.tasks)))

        # Setup for a task-conditional setting
        model_params = config['model']['parameters']
        if 'common_mt_params' in model_params:
            self.task_conditional = not model_params['common_mt_params']
        else:
            self.task_conditional = False

        # Setup optimizers
        optimizer_config = config['optimizer']
        optimizer_cls = get_optimizer(optimizer_config['algorithm'])
        model_params = get_params(self.model, optimizer_config['parameters']['lr'], len(self.tasks),
                                  self.task_conditional, self.tasks)
        self.optimizer = optimizer_cls(model_params, **optimizer_config['parameters'])

        # Setup schedulers
        scheduler_config = config['scheduler']
        scheduler_cls = get_scheduler(scheduler_config['lr_policy'])
        self.scheduler = scheduler_cls(self.optimizer, **scheduler_config['parameters'])

        # Setup loss function
        losses_config = config['loss']
        self.criterions = get_loss_functions(self.tasks, losses_config)

        # Initialise performance meters
        self.best_val_loss = 1e9
        self.train_loss = {}
        self.val_loss = {}
        for task in self.tasks:
            self.train_loss[task] = get_running_meter()
            self.val_loss[task] = get_running_meter()

        # Initialize img logging for visualization
        self.img_logging = get_img_logging(dataset_name, self.tasks)
        self.pred_decoder = get_pred_decoder(dataset_name, self.tasks)

    def resume(self, checkpoint_dir, loader, gpu_info):
        """Resume training process

        Args:
            checkpoint_dir (str): Path to checkpoint to resume from
            loader (object): Dataloader to get current performance of model
            gpu_info(dict): Dictionary with required GPU information
        Returns:
            iterations (int): Current iteration to resume from
        """
        # Get loss of best model
        best_model_name = get_best_model(checkpoint_dir, "model")
        best_model = torch.load(best_model_name)
        self.model.load_state_dict(best_model['state_dict'])
        self.evaluate_model(loader, gpu_info)
        self.best_val_loss = self.get_val_loss()['loss']

        # Load model
        last_model_name = get_last_model(checkpoint_dir, "model")
        last_model = torch.load(last_model_name)
        self.model.load_state_dict(last_model['state_dict'])
        iterations = int(last_model['iteration'])

        # Load optimizer
        last_optimizer = torch.load(os.path.join(checkpoint_dir, 'optimizer.pth'))
        self.optimizer.load_state_dict(last_optimizer['opt'])
        self.scheduler.last_epoch = iterations

        print('Resume from iteration %d' % iterations)
        return iterations

    def load_last_model(self, checkpoint_dir):
        """Load best model

        Args:
            checkpoint_dir (str): Path to checkpoints to load from
        """
        # Get best model
        best_model_name = get_last_model(checkpoint_dir, "model")
        best_model = torch.load(best_model_name)
        self.model.load_state_dict(best_model['state_dict'])
        print('Last model loaded')

    def get_loss(self, outputs, labels, tasks, train=True):
        """Iterate over the different outputs and merge the losses

        Args:
            outputs (dic): Dictionary of magnitude equal to tasks
            labels (dic): Dictionary of magnitude equal to tasks
            tasks: tasks
            train (bool)
        Returns:
            loss: Merged losses
        """
        if not isinstance(train, bool):
            raise ValueError('train input must be boolean')

        batch_size = outputs[tasks[0]].size(0)
        weighted_loss = 0.0
        for ind, task in enumerate(tasks):
            # Compute loss
            current_loss = self.tasks_weighting[task] * self.criterions[task](outputs[task], labels[task])
            weighted_loss += current_loss

            # Store loss where appropriate
            if train:
                self.train_loss[task].update(current_loss.item(), batch_size)
            else:
                self.val_loss[task].update(current_loss.item(), batch_size)
        return weighted_loss

    def update_model(self, samples):
        """Forward pass and update model

        Args:
            samples (tensor): Input/Targets dictionary
        """
        # Initialise for training
        self.train()
        if self.task_conditional:
            for task in self.tasks:
                self.optimizer.zero_grad()

                # Forward propagation and get performance
                input_dic = {'tensor': samples['image'],
                             'task': task}
                outputs = self.forward(input_dic)
                loss = self.get_loss(outputs, samples['labels'], [task])

                # Back propagation and update model
                loss.backward()
                self.optimizer.step()
        else:
            self.optimizer.zero_grad()

            input_dic = {'tensor': samples['image'],
                         'task': None}

            # Forward propagation and get performance
            outputs = self.forward(input_dic)
            loss = self.get_loss(outputs, samples['labels'], self.tasks)

            # Back propagation and update model
            loss.backward()
            self.optimizer.step()
        self.scheduler.step()

    def evaluate_model(self, loader, gpu_info, save_dir=''):
        """Evaluate current model

        Args:
            loader (object): Dataloader to get current performance of model
            gpu_info(dict): Dictionary with required GPU information
            save_dir(bool): Dir to save predictions
        """
        # Prepare for evaluation
        self.eval()
        with torch.no_grad():
            for it, (samples) in enumerate(loader):
                samples = dict_to_device(samples, gpu_info)

                if self.task_conditional:
                    outputs = {}
                    for task in self.tasks:
                        self.optimizer.zero_grad()

                        # Forward propagation and get performance
                        input_dic = {'tensor': samples['image'],
                                     'task': task}
                        outputs.update(self.forward(input_dic))
                else:
                    self.optimizer.zero_grad()
                    input_dic = {'tensor': samples['image'],
                                 'task': None}

                    # Forward propagation and get performance
                    outputs = self.forward(input_dic)

                _ = self.get_loss(outputs, samples['labels'], self.tasks, train=False)

                if save_dir != '':
                    save_img(samples, outputs, self.tasks_weighting, self.pred_decoder, save_dir)

    def get_train_loss(self):
        """Get statistics for training

        Returns:
            statistics (dict): Training statistics
        """
        statistics = {'loss': 0.0}
        for ind, (task, weight) in enumerate(self.tasks_weighting.items()):
            statistics['loss'] += self.train_loss[task].value
            statistics[task] = self.train_loss[task].value
            self.train_loss[task].reset()
        return statistics

    def get_val_loss(self):
        """Get statistics for validation

        Returns:
            statistics (dict): Validation statistics
        """
        statistics = {'loss': 0.0}
        for ind, (task, weight) in enumerate(self.tasks_weighting.items()):
            statistics['loss'] += self.val_loss[task].avg
            statistics[task] = self.val_loss[task].avg
            self.val_loss[task].reset()
        return statistics

    def save(self, checkpoint_dir, iterations, model_statistics):
        """Save training process

        Args:
            checkpoint_dir (str): Path to checkpoint to resume from
            iterations (int): Current iteration
            model_statistics(dict): Statistics to compare too with current best model
        Returns:
        """
        model_name = os.path.join(checkpoint_dir, 'model_%08d.pth' % (iterations + 1))
        torch.save({'state_dict': self.model.state_dict(),
                    'iteration': (iterations)}, model_name)
        shutil.copy(model_name, os.path.join(checkpoint_dir, 'model_last.pth'))

        opt_name = os.path.join(checkpoint_dir, 'optimizer.pth')
        torch.save({'opt': self.optimizer.state_dict()}, opt_name)

        if self.best_val_loss > model_statistics['loss']:
            self.best_val_loss = model_statistics['loss']
            shutil.copy(model_name, os.path.join(checkpoint_dir, 'model_best.pth'))
            print("=> Best model outperformed. Checkpoint saved")
            print("Average loss: {:.{}f}".format(model_statistics['loss'], 10))

    def img_visualization(self, loader, gpu_info, iteration, writer):
        """Visualize the performance of the current model

        Args:
            loader (object): Dataloader to get current performance of model
            gpu_info(dict): Dictionary with required GPU information
            iteration (int): Current iteration
            writer (object): Writer to log new image too
        """
        if self.img_logging is not None:
            # Prepare for evaluation
            self.eval()
            with torch.no_grad():
                for it, (samples) in enumerate(loader):
                    samples = dict_to_device(samples, gpu_info)

                    # Forward propagation and get performance
                    if self.task_conditional:
                        outputs = {}
                        for task in self.tasks:
                            self.optimizer.zero_grad()

                            # Forward propagation and get performance
                            input_dic = {'tensor': samples['image'],
                                         'task': task}
                            outputs.update(self.forward(input_dic))
                    else:
                        self.optimizer.zero_grad()
                        input_dic = {'tensor': samples['image'],
                                     'task': None}

                        # Forward propagation and get performance
                        outputs = self.forward(input_dic)

                    plot_dictionary = dic_for_img_vis(samples, outputs, self.tasks_weighting)

                    self.img_logging.merge_img_labels(plot_dictionary)
                self.img_logging.log_imgs(writer, iteration)

    def forward(self, x):
        return self.model(x)
