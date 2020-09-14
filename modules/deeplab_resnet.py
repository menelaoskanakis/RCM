import os
import numpy as np
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

from copy import deepcopy
from torch.nn import init
from torch.nn import functional as F
from modules.pyramid_pooling import AtrousSpatialPyramidPoolingModule
from collections import OrderedDict
from modules.layers import RCMConv2d, Conv_BatchNorm, MTBatchNorm2d, STMaxPool2d, STReLU, \
    RAConv2d, MTConv2d
from util import dataset_model_info
from modules.conv_bn_pairs import conv_bn_pairs

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
}


def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     dilation=dilation, padding=dilation, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, tasks=None, conv_layer='RCM',
                 NFF=False, common_mt_params=True, train_enc_norm_layers=False, train_enc_conv_layers=False):
        super(BasicBlock, self).__init__()
        """
        Creates a BasicBlock with either RCM, RA, or standard Conv2D

        Args:
            tasks(list): List of tasks
            RCM(bool): Set to True to reparameterize the Conv unit to two convolutions (True=RCM, False=Stand. Conv2D)
            NFF(bool): Set to True to activate NFF on RCM
            common_mt_params (bool): Set to True to have a single optimizable modulator for all tasks 
        """
        print('BasicBlock Conv1')
        if 'RA' in conv_layer:
            self.conv1 = RAConv2d(inplanes, planes, kernel_size=3, stride=stride, padding=dilation, bias=False,
                                  tasks=tasks, dilation=dilation, common_mt_params=common_mt_params, RA=conv_layer[3:])
        else:
            self.conv1 = RCMConv2d(inplanes, planes, kernel_size=3, stride=stride, padding=dilation, bias=False,
                                   tasks=tasks, dilation=dilation, conv_layer=conv_layer, NFF=NFF,
                                   common_mt_params=common_mt_params)
        self.bn1 = MTBatchNorm2d(planes, tasks=tasks, common_mt_params=common_mt_params)

        for i in self.bn1.parameters():
            i.requires_grad = train_enc_norm_layers
        if not train_enc_conv_layers:
            for i in self.conv1.parameters():
                i.requires_grad = train_enc_conv_layers

        self.relu = STReLU(inplace=True)

        print('BasicBlock Conv2')
        if 'RA' in conv_layer:
            self.conv2 = RAConv2d(planes, planes, kernel_size=3, stride=1, padding=dilation, bias=False,
                                  tasks=tasks, dilation=dilation, common_mt_params=common_mt_params, RA=conv_layer[3:])
        else:
            self.conv2 = RCMConv2d(planes, planes, kernel_size=3, stride=1, padding=dilation, bias=False,
                                   tasks=tasks, dilation=dilation, conv_layer=conv_layer, NFF=NFF,
                                   common_mt_params=common_mt_params)
        self.bn2 = MTBatchNorm2d(planes, tasks=tasks, common_mt_params=common_mt_params)

        for i in self.bn2.parameters():
            i.requires_grad = train_enc_norm_layers
        if not train_enc_conv_layers:
            for i in self.conv2.parameters():
                i.requires_grad = train_enc_conv_layers

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out['tensor'] += residual['tensor']
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, dataset, tasks, nInputChannels=3, pyramid_pooling="atrous-v3",
                 output_stride=16, decoder=True, train_enc_norm_layers=False, train_enc_conv_layers=False,
                 conv_layer='RCM', NFF=False, common_mt_params=True):
        super(ResNet, self).__init__()

        print("Constructing ResNet model...")
        print("Output stride: {}".format(output_stride))
        print("Number of tasks: {}".format(len(tasks)))

        if conv_layer not in ['RA_parallel', 'RA_series', 'RCM', 'Conv']:
            raise ValueError('Not a valid conv layer, please choose from: RA_parallel, RA_series, RCM, Conv')

        self.active_tasks = OrderedDict()
        dataset_dic = dataset_model_info(dataset)
        for key, value in dataset_dic.items():
            if key in tasks:
                self.active_tasks[key] = value

        # Define parameters for DeepLab
        v3_atrous_rates = [6, 12, 18]
        if output_stride == 8:
            dilations = (2, 4)
            strides = (2, 2, 2, 1, 1)
            v3_atrous_rates = [x * 2 for x in v3_atrous_rates]
        elif output_stride == 16:
            dilations = (1, 2)
            strides = (2, 2, 2, 2, 1)
        else:
            raise ValueError('Choose between output_stride 8 and 16')

        self.inplanes = 64
        self.pyramid_pooling = pyramid_pooling
        self.decoder = decoder
        self.bnorm = nn.BatchNorm2d

        print('Conv1:')
        if 'RA' in conv_layer:
            self.conv1 = RAConv2d(nInputChannels, 64, kernel_size=7, stride=strides[0], padding=3, bias=False,
                                  tasks=tasks, common_mt_params=common_mt_params, RA=conv_layer[3:])
        else:
            self.conv1 = RCMConv2d(nInputChannels, 64, kernel_size=7, stride=strides[0], padding=3, bias=False,
                                   tasks=tasks, conv_layer=conv_layer, NFF=NFF, common_mt_params=common_mt_params)
        self.bn1 = MTBatchNorm2d(64, tasks=tasks, common_mt_params=common_mt_params)

        for i in self.bn1.parameters():
            i.requires_grad = train_enc_norm_layers
        if not train_enc_conv_layers:
            for i in self.conv1.parameters():
                i.requires_grad = train_enc_conv_layers

        self.relu = STReLU(inplace=True)
        self.maxpool = STMaxPool2d(kernel_size=3, stride=strides[1], padding=1, ceil_mode=False)

        print('Layer1:')
        self.layer1 = self._make_layer(block, 64, layers[0], tasks=tasks, conv_layer=conv_layer, NFF=NFF,
                                       common_mt_params=common_mt_params, train_enc_norm_layers=train_enc_norm_layers,
                                       train_enc_conv_layers=train_enc_conv_layers)
        print('Layer2:')
        self.layer2 = self._make_layer(block, 128, layers[1], tasks=tasks, stride=strides[2], conv_layer=conv_layer,
                                       NFF=NFF, common_mt_params=common_mt_params,
                                       train_enc_norm_layers=train_enc_norm_layers,
                                       train_enc_conv_layers=train_enc_conv_layers)
        print('Layer3:')
        self.layer3 = self._make_layer(block, 256, layers[2], tasks=tasks, stride=strides[3], dilation=dilations[0],
                                       conv_layer=conv_layer, NFF=NFF, common_mt_params=common_mt_params,
                                       train_enc_norm_layers=train_enc_norm_layers,
                                       train_enc_conv_layers=train_enc_conv_layers)
        print('Layer4:')
        self.layer4 = self._make_layer(block, 512, layers[3], tasks=tasks, stride=strides[4], dilation=dilations[1],
                                       conv_layer=conv_layer, NFF=NFF, common_mt_params=common_mt_params,
                                       train_enc_norm_layers=train_enc_norm_layers,
                                       train_enc_conv_layers=train_enc_conv_layers)

        if block == BasicBlock:
            in_f, out_f = 512, 128
            low_level_dim = 64
        else:
            raise ValueError('Pipeline currently only supports architectures with BasicBlock units.')

        if decoder:
            print('Using decoder')
            if pyramid_pooling == 'atrous-v3':
                print('Initializing pyramid pooling: Atrous pyramid with global features (Deeplab-v3)')
                out_f_pyramid = 256
                self.layer5 = nn.ModuleDict({task: AtrousSpatialPyramidPoolingModule(depth=out_f_pyramid,
                                                                                     dilation_series=v3_atrous_rates,
                                                                                     in_f=in_f)
                                             for task in tasks})
            else:
                raise NotImplementedError("Only 'atrous-v3' pooling layers is supported")

            NormModule = self.bnorm
            kwargs_low = {"num_features": 48}
            kwargs_out = {"num_features": 256}

            self.low_level_reduce = nn.ModuleDict({task: nn.Sequential(
                nn.Conv2d(low_level_dim, 48, kernel_size=1, bias=False),
                NormModule(**kwargs_low),
                nn.ReLU(inplace=True)
            )
                for task in tasks})

            self.concat = nn.ModuleDict({task: nn.Sequential(
                conv3x3(out_f_pyramid + 48, 256),
                NormModule(**kwargs_out),
                nn.ReLU(inplace=True),
                conv3x3(256, 256),
                NormModule(**kwargs_out),
                nn.ReLU(inplace=True)
            )
                for task in tasks})

            self.predict = nn.ModuleDict({task: self._make_prediction_layer(task) for task in tasks})
        else:
            raise NotImplementedError('Only supports the use of decoders')

        # Initialize weights
        self._initialize_weights()

        # Check if batchnorm parameters are trainable
        self._verify_bnorm_params()

    def _make_prediction_layer(self, task):
        return nn.Conv2d(256, self.active_tasks[task]['out_dim'], kernel_size=1, bias=True)

    def _make_layer(self, block, planes, blocks, tasks=None, stride=1, dilation=1, conv_layer='RCM', NFF=False,
                    common_mt_params=True, train_enc_norm_layers=False, train_enc_conv_layers=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or dilation == 2 or dilation == 4:
            print('downsample')
            downsample = Conv_BatchNorm(conv_layer=MTConv2d,
                                        conv_kwargs={'in_planes': self.inplanes,
                                                     'out_planes': planes * block.expansion,
                                                     'kernel_size': 1,
                                                     'stride': stride,
                                                     'bias': False,
                                                     'tasks': tasks,
                                                     'common_mt_params': common_mt_params},
                                        train_enc_conv_layers=train_enc_conv_layers,
                                        bnorm_layer=MTBatchNorm2d,
                                        norm_kwargs={'num_features': planes * block.expansion,
                                                     'tasks': tasks,
                                                     'common_mt_params': common_mt_params},
                                        train_enc_norm_layers=train_enc_norm_layers
                                        )

        layers = []
        print('Residual Block 0')
        layers.append(block(self.inplanes, planes, stride, tasks=tasks, dilation=dilation, downsample=downsample,
                            conv_layer=conv_layer, NFF=NFF, common_mt_params=common_mt_params,
                            train_enc_norm_layers=train_enc_norm_layers, train_enc_conv_layers=train_enc_conv_layers))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            print('Residual Block {}'.format(str(i)))
            layers.append(block(self.inplanes, planes, tasks=tasks, dilation=dilation, conv_layer=conv_layer, NFF=NFF,
                                common_mt_params=common_mt_params, train_enc_norm_layers=train_enc_norm_layers,
                                train_enc_conv_layers=train_enc_conv_layers))

        return nn.Sequential(*layers)

    def _verify_bnorm_params(self):
        verify_trainable = True
        a = 0
        for x in self.modules():
            if isinstance(x, nn.BatchNorm2d):
                for y in x.parameters():
                    verify_trainable = (verify_trainable and y.requires_grad)
                a += isinstance(x, nn.BatchNorm2d)

        print("\nVerification: Trainable batchnorm parameters? Answer: {}\n".format(verify_trainable))
        print("bnorm layers: {}".format(a))

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, self.bnorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight)
                m.bias.data.zero_()

    def forward(self, x):
        output = {}
        h, w = x['tensor'].shape[2:]
        task = x['task']

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        if self.decoder:
            x_low = x['tensor']
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if task is not None:
            x = self.layer5[task](x['tensor'])
            if self.decoder:
                x = F.interpolate(x, size=(x_low.shape[2], x_low.shape[3]),
                                  mode='bilinear', align_corners=False)

                x_low = self.low_level_reduce[task](x_low)
                x = torch.cat([x, x_low], dim=1)
                x = self.concat[task](x)
                pred = self.predict[task](x)
                output = {task: F.interpolate(pred, size=(h, w), mode='bilinear', align_corners=False)}
                if self.active_tasks[task]['normalize']:
                    output[task] = F.normalize(output[task], p=2, dim=1)
            else:
                NotImplementedError('Only supports the use of decoders')
        else:
            for task in self.active_tasks:
                x_dec = self.layer5[task](x['tensor'])
                if self.decoder:
                    x_dec = F.interpolate(x_dec, size=(x_low.shape[2], x_low.shape[3]),
                                          mode='bilinear', align_corners=False)
                    x_dec_low = self.low_level_reduce[task](x_low)
                    x_dec = torch.cat([x_dec, x_dec_low], dim=1)
                    x_dec = self.concat[task](x_dec)
                    pred = self.predict[task](x_dec)

                    output[task] = F.interpolate(pred, size=(h, w), mode='bilinear', align_corners=False)
                    if self.active_tasks[task]['normalize']:
                        output[task] = F.normalize(output[task], p=2, dim=1)
                else:
                    NotImplementedError('Only supports the use of decoders')

        return output


def get_state_dict(model_dir, model_name, pretrained_architecture):
    if pretrained_architecture in ['Conv', 'RC_RI']:
        checkpoint = model_zoo.load_url(model_urls[model_name])
    else:
        new_model_name = "{}_{}.pth".format(model_name, pretrained_architecture)
        # Load checkpoint
        checkpoint = torch.load(
            os.path.join(model_dir, new_model_name), map_location=lambda storage, loc: storage)
        checkpoint = checkpoint['state_dict']

    # Handle DataParallel
    if 'module.' in list(checkpoint.keys())[0]:
        state_dict = OrderedDict()
        for k, v in checkpoint.items():
            name = k.replace('module.', '')
            state_dict[name] = v
        return state_dict
    else:
        return checkpoint


def generate_new_weights(name, weights, decomposition_path='./results/activations'):
    new_name = name.replace('.', '_')

    if 'downsample_conv_conv' in new_name:
        new_name = new_name.replace('downsample_conv_conv', 'downsample_0')

    decomposition = torch.from_numpy(np.load(os.path.join(os.path.join(decomposition_path, new_name), 'components.npy')))

    y_mean = torch.from_numpy(np.load(os.path.join(os.path.join(decomposition_path, new_name), 'mean.npy')))
    y_mean = y_mean.unsqueeze(1)
    M = torch.mm(decomposition, decomposition.transpose(-2, -1))

    Ut = decomposition.transpose(-2, -1)
    U = decomposition
    bias = (y_mean - torch.mm(M, y_mean)).squeeze()
    return weights, Ut, U, bias


def create_decompositions(checkpoint, decomposition_path, pretrained_architecture):
    # Handle DataParallel
    state_dict = OrderedDict()
    state_dict_bias = OrderedDict()
    for name, v in checkpoint.items():
        if 'bn' not in name and 'downsample' not in name:
            if 'fc' in name:
                state_dict[name] = v
                continue
            name = name.replace('.weight', '')
            W, Ut, U, bias = generate_new_weights(name, v, decomposition_path=decomposition_path)

            weights_out, weights_in, weights_h, weights_w = W.size()
            state_dict[name + '.Ws.weight'] = torch.mm(Ut, W.view(weights_out, -1)).view(-1, weights_in,
                                                                                         weights_h, weights_w)
            state_dict[name + '.Wt.weight'] = U.unsqueeze(2).unsqueeze(3)
            state_dict_bias[conv_bn_pairs[name]] = bias

        else:
            state_dict[name] = v

    for bias_name, value in state_dict_bias.items():
        state_dict[bias_name] = state_dict[bias_name] + value
    return state_dict


def rename_weights(checkpoint, pretrained_architecture):
    state_dict = OrderedDict()
    for name, v in checkpoint.items():
        if 'bn' not in name and 'downsample' not in name:
            if 'fc' in name:
                state_dict[name] = v
                continue
            name = name.replace('.weight', '.Ws.weight')
        state_dict[name] = v
    return state_dict


def create_nff_layer(checkpoint, pretrained_architecture):
    # Handle DataParallel
    state_dict = OrderedDict()
    for name, v in checkpoint.items():
        if 'Wt.weight' in name:
            g = torch.norm(v.squeeze(), p=2, dim=1).unsqueeze(1).unsqueeze(2).unsqueeze(3)
            state_dict[name + '_v'] = v
            state_dict[name + '_g'] = g
        else:
            state_dict[name] = v
    return state_dict


def adjust_old_to_new_dict(model, pretrained_dict, tasks, pretrained_architecture):
    current_state_dic = model.state_dict()

    new_dict = {}
    for k, v in current_state_dic.items():
        name = deepcopy(k)
        for task in tasks:
            name = name.replace(task + '.', '')

        if any(x in name for x in ['bn1', 'bn2', 'bn3']):
            name = name.replace('.bnorm', '')

        if 'conv' in name and '.bnorm.bnorm' in name:
            name = name.replace('.bnorm.bnorm', '.bn')

        if name in pretrained_dict:
            new_dict[k] = pretrained_dict[name]
        else:
            if any(part in name for part in ['predict', 'concat', 'low_level_reduce', 'layer5',
                                             'num_batches_tracked']):
                new_dict[k] = v
            else:
                raise ValueError('Using random value instead of ImageNet initialization for {}'.format(k))
    return new_dict


def adjust_downsample_dic(pretrained_dict):
    state_dict = OrderedDict()
    for name, v in pretrained_dict.items():
        if 'downsample.0' in name:
            name = name.replace('downsample.0', 'downsample.conv.conv')
        elif 'downsample.1' in name:
            name = name.replace('downsample.1', 'downsample.bnorm.bnorm')
        state_dict[name] = v

    return state_dict


def resnet18(tasks, pretrained_architecture=None, train_enc_norm_layers=True, train_enc_conv_layers=True,
             conv_layer='RCM', NFF=False, common_mt_params=True, model_dir=None, decomp_path=None, **kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], tasks=tasks, train_enc_norm_layers=train_enc_norm_layers,
                   train_enc_conv_layers=train_enc_conv_layers, conv_layer=conv_layer, NFF=NFF,
                   common_mt_params=common_mt_params, **kwargs)

    if pretrained_architecture is None:
        print('Training from scratch')
        return model
    elif pretrained_architecture not in ['RA_series', 'RA_parallel', 'Conv', 'RC', 'RC_RI']:
        raise ValueError('Pretrained architecture {} has not been implemented'.format(pretrained_architecture))
    elif conv_layer == 'RCM':
        if not (pretrained_architecture in ['RC', 'RC_RI']):
            raise ValueError("Pretrained model options for RCM are 'RC' or 'RC_RI', not {}"
                             .format(pretrained_architecture))
    else:
        if not (pretrained_architecture == conv_layer):
            raise ValueError("Pretrained model for {} is {}, and not {}"
                             .format(conv_layer, conv_layer, pretrained_architecture))

    model_name = 'resnet18'
    print('Loading Imagenet {} for {} initialization'.format(model_name, pretrained_architecture))
    state_dict = get_state_dict(model_dir, model_name, pretrained_architecture)
    state_dict = adjust_downsample_dic(state_dict)

    if pretrained_architecture == 'RC_RI':
        state_dict = create_decompositions(state_dict, decomp_path, pretrained_architecture)
    elif pretrained_architecture == 'Conv':
        state_dict = rename_weights(state_dict, pretrained_architecture)

    if conv_layer != 'RCM' and NFF:
        raise ValueError("Only RCM architecture supports NFF. You are using {}".format(conv_layer))
    elif conv_layer == 'RCM' and NFF:
        state_dict = create_nff_layer(state_dict, pretrained_architecture)

    new_dict = adjust_old_to_new_dict(model, state_dict, tasks, pretrained_architecture)
    model.load_state_dict(new_dict)

    return model


def resnet34(tasks, pretrained_architecture=None, train_enc_norm_layers=True, train_enc_conv_layers=True,
             conv_layer='RCM', NFF=False, common_mt_params=True, model_dir=None, decomp_path=None, **kwargs):
    """Constructs a ResNet-34 model.
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], tasks=tasks, train_enc_norm_layers=train_enc_norm_layers,
                   train_enc_conv_layers=train_enc_conv_layers, conv_layer=conv_layer, NFF=NFF,
                   common_mt_params=common_mt_params, **kwargs)

    if pretrained_architecture is None:
        print('Training from scratch')
        return model
    elif pretrained_architecture not in ['Conv', 'RC_RI']:
        raise ValueError('Pretrained architecture {} has not been implemented'.format(pretrained_architecture))
    elif conv_layer == 'RCM':
        if not (pretrained_architecture in ['RC_RI']):
            raise ValueError("Pretrained model option for RCM is 'RC_RI', not {}"
                             .format(pretrained_architecture))
    else:
        if not (pretrained_architecture == conv_layer):
            raise ValueError("Pretrained model for {} is {}, and not {}"
                             .format(conv_layer, conv_layer, pretrained_architecture))

    model_name = 'resnet34'
    print('Loading Imagenet {} for {} initialization'.format(model_name, pretrained_architecture))
    state_dict = get_state_dict(model_dir, model_name, pretrained_architecture)
    state_dict = adjust_downsample_dic(state_dict)

    if pretrained_architecture == 'RC_RI':
        state_dict = create_decompositions(state_dict, decomp_path, pretrained_architecture)
    elif pretrained_architecture == 'Conv':
        state_dict = rename_weights(state_dict, pretrained_architecture)

    if conv_layer != 'RCM' and NFF:
        raise ValueError("Only RCM architecture supports NFF. You are using {}".format(conv_layer))
    elif conv_layer == 'RCM' and NFF:
        state_dict = create_nff_layer(state_dict, pretrained_architecture)

    new_dict = adjust_old_to_new_dict(model, state_dict, tasks, pretrained_architecture)
    model.load_state_dict(new_dict)

    return model
