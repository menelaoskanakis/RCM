import torch.nn.functional as F
import numpy as np
import os

from torch.nn.modules.utils import _pair
from torch.nn.modules.conv import _ConvNd


def create_dir(directory):
    """Create directory if it does not exist

    Args:
        directory(str): Directory to create
    """
    if not os.path.exists(directory):
        print("Creating directory: {}".format(dir))
        os.makedirs(directory)


class ResponseConv2d(_ConvNd):
    """

    """
    def __init__(self, in_channels, out_channels, kernel_size, activation_path=None, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(ResponseConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode)
        if activation_path is None:
            raise ValueError('activation_path not defined')
        self.activation_path = activation_path
        create_dir(self.activation_path)
        self.out_channels = out_channels
        self.ind = 0

    def forward(self, input):
        response = F.conv2d(input, self.weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)
        outputs = response.permute(1, 0, 2, 3).contiguous().view(self.out_channels, -1).cpu().numpy()
        np.save(os.path.join(self.activation_path, 'batch_' + str(self.ind)), outputs)
        self.ind += 1
        return response


