import torch.nn as nn
from torch.nn import functional as F
import torch


class AtrousSpatialPyramidPoolingModule(nn.Module):
    """
    Atrous Spatial Pyramid Pooling Module (DeepLab-v3+)
    """
    def __init__(self, dilation_series=[6, 12, 18], depth=256, in_f=2048, cardinality=1):
        super(AtrousSpatialPyramidPoolingModule, self).__init__()

        padding_series = dilation_series
        self.conv2d_list = nn.ModuleList()
        self.bnorm = nn.BatchNorm2d
        NormModule = self.bnorm
        kwargs = {"num_features": depth}

        # 1x1 convolution
        self.conv2d_list.append(nn.Sequential(
            nn.Conv2d(in_f, depth, kernel_size=1, stride=1, bias=False),
            NormModule(**kwargs),
            nn.ReLU(inplace=True)))

        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(nn.Sequential(
                nn.Conv2d(in_f, depth, kernel_size=3, stride=1, padding=padding,
                          dilation=dilation, bias=False, groups=cardinality),
                NormModule(**kwargs),
                nn.ReLU(inplace=True)))

        # Global features
        self.conv2d_list.append(nn.Sequential(nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                                              nn.Conv2d(in_f, depth, kernel_size=1, stride=1, bias=False),
                                              NormModule(**kwargs),
                                              nn.ReLU(inplace=True)))

        self.conv2d_final = nn.Sequential(
            nn.Conv2d(depth * 5, depth, kernel_size=1, stride=1, bias=False),
            NormModule(**kwargs),
            nn.ReLU(inplace=True))

        for m in self.conv2d_list:
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, self.bnorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        for m in self.conv2d_final:
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, self.bnorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        h, w = x.size(2), x.size(3)

        interm = []
        for i in range(len(self.conv2d_list)):
            interm.append(self.conv2d_list[i](x))

        # Upsample the global features
        interm[-1] = F.interpolate(input=interm[-1], size=(h, w), mode='bilinear', align_corners=False)

        # Concatenate the parallel streams
        out = torch.cat(interm, dim=1)

        # Final convolutional layer of the classifier
        out = self.conv2d_final(out)

        return out
