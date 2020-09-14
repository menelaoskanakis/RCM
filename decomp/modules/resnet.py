import math
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import os

from .layers import ResponseConv2d

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

__all__ = ['ResNet', 'resnet18', 'resnet34']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
}


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, activation_path=None):
        super(BasicBlock, self).__init__()

        conv_activation_path = activation_path + '_conv1'
        self.conv1 = ResponseConv2d(inplanes, planes, kernel_size=3, activation_path=conv_activation_path,
                                    stride=stride, padding=1, bias=False)

        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        conv_activation_path = activation_path + '_conv2'
        self.conv2 = ResponseConv2d(planes, planes, kernel_size=3, activation_path=conv_activation_path,
                                    stride=1, padding=1, bias=False)

        self.bn2 = nn.BatchNorm2d(planes)
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

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, activation_root=None, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()

        activation_path = os.path.join(activation_root, 'conv1')
        self.conv1 = ResponseConv2d(3, 64, kernel_size=7, activation_path=activation_path, stride=2, padding=3, bias=False)

        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        activation_path = os.path.join(activation_root, 'layer1_')
        self.layer1 = self._make_layer(block, 64, layers[0], activation_path=activation_path)

        activation_path = os.path.join(activation_root, 'layer2_')
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, activation_path=activation_path)

        activation_path = os.path.join(activation_root, 'layer3_')
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, activation_path=activation_path)

        activation_path = os.path.join(activation_root, 'layer4_')
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, activation_path=activation_path)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, activation_path=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion,
                                        kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes * block.expansion),
                )

        layers = []

        block_activation_path = activation_path + str(0)
        layers.append(block(self.inplanes, planes, stride=stride, downsample=downsample, activation_path=block_activation_path))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            block_activation_path = activation_path + str(i)
            layers.append(block(self.inplanes, planes, activation_path=block_activation_path))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def get_state_dict(model_name):
    # Load checkpoint
    if model_name in model_urls:
        checkpoint = model_zoo.load_url(model_urls[model_name])
    else:
        raise ValueError("Model defined does not exist!")
    return checkpoint


def resnet18(activation_root=None, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model_name = 'resnet18'
    activation_root = os.path.join(activation_root, model_name)
    model = ResNet(BasicBlock, [2, 2, 2, 2], activation_root=activation_root, **kwargs)

    state_dict = get_state_dict(model_name)
    model.load_state_dict(state_dict)

    return model


def resnet34(activation_root=None, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model_name = 'resnet34'
    activation_root = os.path.join(activation_root, model_name)
    model = ResNet(BasicBlock, [3, 4, 6, 3], activation_root=activation_root, **kwargs)

    state_dict = get_state_dict(model_name)
    model.load_state_dict(state_dict)

    return model
