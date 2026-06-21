import torch
import torch.nn as nn
import torch.nn.functional as F

class FusedCrossEntropyLoss(nn.Module):
    def __init__(
        self,
        ignore_index=-100,
        reduction="mean",
        label_smoothing=0.0,
        inplace_backward=True,
        process_group=None,
    ):
        super().__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, input, target):
        if len(input.shape) == 3:
            input = input.view(-1, input.size(-1))
            target = target.view(-1)
            
        return F.cross_entropy(
            input,
            target,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
            label_smoothing=self.label_smoothing
        )
