import torch
from torch import nn, Tensor
import numpy as np
import torch.nn.functional as F


class RobustCrossEntropyLoss(nn.CrossEntropyLoss):
    """
    this is just a compatibility layer because my target tensor is float and has an extra dimension
    """

    def forward(self, input: Tensor, target: Tensor, w) -> Tensor:
        if len(target.shape) == len(input.shape):
            assert target.shape[1] == 1
            target = target[:, 0]
        # todo modify CE loss
        weight = w
        ce_score = 0.
        for b in range(input.shape[0]):
            ce_score += F.cross_entropy(input[b:b+1], target[b:b+1], weight=weight[b],
                                             ignore_index=self.ignore_index, reduction=self.reduction,
                                             label_smoothing=self.label_smoothing)
        return ce_score / input.shape[0]
