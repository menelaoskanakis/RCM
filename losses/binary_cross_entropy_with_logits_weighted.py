import torch.nn as nn
import torch


class BCEWithLogitsLossWeighted(nn.Module):
    """
    Balanced Cross Entropy Loss with optional ignore regions
    """

    def __init__(self, reduction='mean', average='over_size', positive_weight=None, ignore_index=255):
        super().__init__()
        assert reduction == 'mean'
        self.average = average
        self.positive_weight = positive_weight
        self.ignore_index = ignore_index

    def forward(self, output, label):
        output = output.squeeze()
        label = label.squeeze()

        assert (output.size() == label.size())

        mask = torch.le(label, 1.1)
        masked_label = torch.masked_select(label, mask)
        masked_output = torch.masked_select(output, mask)

        # making sure target is binary
        masked_labels = torch.ge(masked_label, 0.5).float()

        # Weighting of the loss, default is HED-style
        if self.positive_weight is None:
            num_labels_neg = torch.sum(1.0 - masked_labels)
            num_total = torch.numel(masked_labels)
            w = num_labels_neg / num_total
        else:
            w = self.positive_weight

        output_gt_zero = torch.ge(masked_output, 0).float()
        loss_val = torch.mul(masked_output, (masked_labels - output_gt_zero)) - torch.log(
            1 + torch.exp(masked_output - 2 * torch.mul(masked_output, output_gt_zero)))

        loss_pos_pix = -torch.mul(masked_labels, loss_val)
        loss_neg_pix = -torch.mul(1.0 - masked_labels, loss_val)

        loss_pos = torch.sum(loss_pos_pix)
        loss_neg = torch.sum(loss_neg_pix)

        final_loss = w * loss_pos + (1 - w) * loss_neg

        if self.average == 'over_size':
            final_loss /= float(torch.numel(masked_label))
        elif self.average == 'over_batch':
            final_loss /= label.size()[0]
        else:
            raise NotImplementedError

        return final_loss
