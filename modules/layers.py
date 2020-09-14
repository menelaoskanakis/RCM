import torch.nn as nn
import torch.nn.functional as F

from torch.nn.modules.utils import _pair
from torch.nn.modules.conv import _ConvNd
from torch.nn.utils.weight_norm import weight_norm


class RCMConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, tasks=None, stride=1, padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros', conv_layer='RCM', NFF=False, common_mt_params=True):
        super(RCMConv2d, self).__init__()
        """
        Create the RCM unit based on Conv2D

        Args:
            tasks(list): List of tasks
            conv_layer(bool): Set to True to reparameterize the Conv unit to two convolutions (True=RCM, False=Stand. Conv2D)
            NFF(bool): Set to True to activate NFF on RCM
            common_mt_params (bool): Set to True to have a single optimizable modulator for multiple tasks 
        """
        # Initial convolution (filter bank in case of RCM)
        self.Ws = STConv2d(in_planes, out_planes, kernel_size, stride=stride, padding=padding, dilation=dilation,
                           groups=groups, bias=bias)

        # Define modulator
        if conv_layer == 'RCM':
            self.RCM_flag = True
        elif conv_layer == 'Conv':
            self.RCM_flag = False
        else:
            raise ValueError('Invalid conv_layer name for RCMConv2d. Choose from: RCM, Conv')

        self.common_mt_params = common_mt_params
        if self.RCM_flag and NFF:
            if self.common_mt_params:
                print('    Using an RCM w/NFF for all tasks')
                self.Wt = weight_norm(STConv2d(out_planes, out_planes, 1, stride=1, groups=groups, bias=bias))
            else:
                print('    Using an RCM w/NFF per task')
                self.Wt = nn.ModuleDict({task: weight_norm(STConv2d(out_planes, out_planes, 1, stride=1,
                                                                    groups=groups, bias=bias)) for task in tasks})
        elif self.RCM_flag:
            if self.common_mt_params:
                print('    Using an RCM wo/NFF for all tasks')
                self.Wt = STConv2d(out_planes, out_planes, 1, stride=1, groups=groups, bias=bias)
            else:
                print('    Using an RCM wo/NFF per task')
                self.Wt = nn.ModuleDict({task: STConv2d(out_planes, out_planes, 1, stride=1,
                                                        groups=groups, bias=bias) for task in tasks})
        else:
            print('    Using a normal convolution')

        # Ws is not trainable if modulator RCM exists
        if self.RCM_flag:
            for i in self.Ws.parameters():
                i.requires_grad = False

    def forward(self, x):
        out = self.Ws(x)

        if self.RCM_flag:
            if self.common_mt_params:
                out = self.Wt(out)
            else:
                out = self.Wt[out['task']](out)
        return out


class RAConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros', RA='parallel', tasks=None, common_mt_params=True):
        super(RAConv2d, self).__init__()
        """
        Create the RA module based on Conv2D

        Args:
            RA(string): Name of residual adapter module
            tasks(list): List of tasks
            common_mt_params (bool): Set to True to have a single optimizable modulator for multiple tasks 
        """

        if RA not in ['series', 'parallel']:
            raise ValueError('Invalid RA adapter {}. Please choose from series/parallel'.format(self.RA))
        else:
            print('Using RA: ' + RA)
            self.RA = RA

        # Initial convolution
        self.Ws = STConv2d(in_planes, out_planes, kernel_size, stride=stride,
                           padding=padding, dilation=dilation, groups=groups,
                           bias=bias)

        self.common_mt_params = common_mt_params
        if self.RA == 'series':
            if self.common_mt_params:
                print('    Using a single Series RA for all tasks')
                self.bnorm = MTBatchNorm2d(out_planes, tasks=tasks, common_mt_params=common_mt_params)
                self.Wt = STConv2d(out_planes, out_planes, 1, stride=1, groups=groups, bias=bias)
            else:
                print('    Using a Series RA per task')
                self.bnorm = MTBatchNorm2d(out_planes, tasks=tasks, common_mt_params=common_mt_params)
                self.Wt = nn.ModuleDict({task: STConv2d(out_planes, out_planes, 1, stride=1,
                                                        groups=groups, bias=bias) for task in tasks})
        elif self.RA == 'parallel':
            if self.common_mt_params:
                print('    Using a single Parallel RA for all tasks')
                self.Wt = STConv2d(in_planes, out_planes, 1, stride=stride, groups=groups, bias=bias)
            else:
                print('    Using a Parallel RA per task')
                self.Wt = nn.ModuleDict({task: STConv2d(in_planes, out_planes, 1, stride=stride,
                                                       groups=groups, bias=bias) for task in tasks})
        else:
            raise ValueError('Invalid RA adapter {}. Please choose from series/parallel'.format(self.RA))

        # Freeze Ws when training RAs
        for i in self.Ws.parameters():
            i.requires_grad = False

    def forward(self, x):
        if self.RA == 'series':
            x = self.Ws(x)
            if self.common_mt_params:
                out = self.bnorm(x)
                out = self.Wt(out)
            else:
                out = self.bnorm(x)
                out = self.Wt[x['task']](out)
            out['tensor'] = x['tensor'] + out['tensor']
        elif self.RA == 'parallel':
            out = self.Ws(x)
            if self.common_mt_params:
                out['tensor'] = self.Wt(x)['tensor'] + out['tensor']
            else:
                out['tensor'] = self.Wt[x['task']](x)['tensor'] + out['tensor']
        else:
            raise ValueError('Not a valid RA')

        return out


class Conv_BatchNorm(nn.Module):
    def __init__(self,
                 conv_layer=None,
                 conv_kwargs=None,
                 train_enc_conv_layers=False,
                 bnorm_layer=None,
                 norm_kwargs=None,
                 train_enc_norm_layers=False):
        super(Conv_BatchNorm, self).__init__()
        """
        Create a unified block comprised of conv and batch norm (task specific/common/RCM)

        Args:
            conv_layer: Desired convolution layer
            conv_kwargs(dic): Desired convolution parameters
            bnorm_layer: Desired batch norm layer
            norm_kwargs(dic): Desired batch norm parameters
        """
        # Processing layer
        self.conv = conv_layer(**conv_kwargs)
        self.bnorm = bnorm_layer(**norm_kwargs)

        # Define whether cond and batch norm parameters are trainable
        for i in self.bnorm.parameters():
            i.requires_grad = train_enc_norm_layers
        if not train_enc_conv_layers:
            for i in self.conv.parameters():
                i.requires_grad = train_enc_conv_layers

    def forward(self, x):
        out = self.conv(x)
        out = self.bnorm(out)
        return out


class MTBatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, track_running_stats=True, tasks=None,
                 common_mt_params=True):
        super(MTBatchNorm2d, self).__init__()
        """
        Create a multi task batch norm (single common one or task specific)

        Args:
            tasks(list): List of tasks
            common_mt_params (bool): Set to True to have a single optimizable modulator for multiple tasks 
        """
        # Specify common or Task specific batch norm
        self.common_mt_params = common_mt_params
        if common_mt_params:
            print('    Using a single Batch Norm')
            self.bnorm = nn.BatchNorm2d(num_features, eps=eps, momentum=momentum,
                                        track_running_stats=track_running_stats)
        else:
            print('    Using a Batch Norm per task')
            self.bnorm = nn.ModuleDict({task: nn.BatchNorm2d(num_features, eps=eps, momentum=momentum,
                                                             track_running_stats=track_running_stats)
                                        for task in tasks})

    def forward(self, input_dic):
        output_dic = {'task': input_dic['task']}
        if self.common_mt_params:
            output_dic['tensor'] = self.bnorm(input_dic['tensor'])
        else:
            output_dic['tensor'] = self.bnorm[input_dic['task']](input_dic['tensor'])
        return output_dic


class MTConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, tasks=None, stride=1, padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros', common_mt_params=True):
        super(MTConv2d, self).__init__()
        """
        Create a multi task conv (single common one or task specific)

        Args:
            tasks(list): List of tasks
            common_mt_params (bool): Set to True to have a single optimizable modulator for multiple tasks 
        """
        # Specify common or Task specific Conv
        self.common_mt_params = common_mt_params
        if common_mt_params:
            print('    Using a single Conv')
            self.conv = STConv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=kernel_size,
                                 stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias,
                                 padding_mode=padding_mode)
        else:
            print('    Using a Conv per task')
            self.conv = nn.ModuleDict({task: STConv2d(in_channels=in_planes, out_channels=out_planes,
                                                      kernel_size=kernel_size, stride=stride, padding=padding,
                                                      dilation=dilation, groups=groups, bias=bias,
                                                      padding_mode=padding_mode)
                                       for task in tasks})

    def forward(self, input_dic):
        if self.common_mt_params:
            output_dic = self.conv(input_dic)
        else:
            output_dic = self.conv[input_dic['task']](input_dic)
        return output_dic


class STReLU(nn.Module):
    def __init__(self, inplace=False):
        super(STReLU, self).__init__()
        """
        Works as the normal Relu, but can operate with the dictionary input setup
        """
        self.relu = nn.ReLU(inplace)

    def forward(self, input_dic):
        output_dic = {'tensor': self.relu(input_dic['tensor']),
                      'task': input_dic['task'],
                      }
        return output_dic


class STConv2d(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(STConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode)
        """
        Works as the normal Conv2d, but can operate with the dictionary input setup
        """

    def forward(self, input_dic):
        output_dic = {'tensor': F.conv2d(input_dic['tensor'], self.weight, self.bias, self.stride,
                                         self.padding, self.dilation, self.groups),
                      'task': input_dic['task'],
                      }
        return output_dic


class STMaxPool2d(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False):
        super(STMaxPool2d, self).__init__()
        """
        Works as the normal MaxPool2d, but can operate with the dictionary input setup
        """
        self.MaxPool2d = nn.MaxPool2d(kernel_size, stride=stride, padding=padding, dilation=dilation, return_indices=return_indices, ceil_mode=ceil_mode)

    def forward(self, input_dic):
        output_dic = {'tensor': self.MaxPool2d(input_dic['tensor']),
                      'task': input_dic['task'],
                      }
        return output_dic
