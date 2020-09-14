import torch.nn as nn


class L1MaskedLossDepth(nn.Module):
    """
    L1 masked loss depth
    """
    def __init__(self):
        super(L1MaskedLossDepth, self).__init__()
        self.criterion = nn.L1Loss(reduction='sum')

    def forward(self, input, target):
        input = input.squeeze()
        target = target.squeeze()

        mask = (target != 0).float()
        loss = self.criterion(input*mask, target*mask)/(mask.sum() + 1e-12)
        return loss