import torch.nn as nn


class PHLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred, target):
        pred = pred.view(-1)
        target = target.float().view(-1)
        return self.bce(pred, target)
