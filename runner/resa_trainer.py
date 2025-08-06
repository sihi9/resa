import torch.nn as nn
import torch
import torch.nn.functional as F

from runner.registry import TRAINER

@TRAINER.register_module
class RESA(nn.Module):
    def __init__(self, cfg):
        super(RESA, self).__init__()
        self.cfg = cfg

        # Binary segmentation loss (includes sigmoid internally)
        self.criterion = nn.BCEWithLogitsLoss().cuda()

    def forward(self, net, batch):
        output = net(batch['img'])  # output['seg'] expected

        pred = output['seg']  # [B, 1, H, W]
        gt = batch['label'].float().unsqueeze(1)  # [B, H, W] -> [B, 1, H, W]

        seg_loss = self.criterion(pred, gt)
        loss = seg_loss * self.cfg.seg_loss_weight

        loss_stats = {'seg_loss': seg_loss}

        return {'loss': loss, 'loss_stats': loss_stats}
