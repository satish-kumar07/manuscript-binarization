# src/losses.py
import torch, torch.nn as nn, torch.nn.functional as F

class BCEDiceLoss(nn.Module):
    def __init__(self, w_bce=0.5, w_dice=0.5, eps=1.0):
        super().__init__(); self.wb, self.wd, self.eps = w_bce, w_dice, eps
    def forward(self, logits, target):
        bce = F.binary_cross_entropy_with_logits(logits, target)
        p = torch.sigmoid(logits)
        inter = (p * target).sum((1, 2, 3))
        dice = 1 - (2 * inter + self.eps) / (p.sum((1, 2, 3)) + target.sum((1, 2, 3)) + self.eps)
        return self.wb * bce + self.wd * dice.mean()
