import torch.nn as nn
import torch


class L1MaskedLossNormals(nn.Module):
    """
    L1 masked loss normals
    """
    def __init__(self):
        super(L1MaskedLossNormals, self).__init__()
        self.criterion = nn.L1Loss(reduction='sum')

    def forward(self, input, target):
        mask = (torch.abs(torch.norm(target, p=2, dim=1)) > 0).unsqueeze(1).repeat(1, 3, 1, 1).float()
        loss = self.criterion(input*mask, target*mask)/(mask.sum() + 1e-12)
        return loss