import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from utils.config import Config
from models.registry import build_net
from datasets import build_dataset

import torch.nn.functional as F


def compute_batch_iou(preds, targets, threshold=0.5, eps=1e-6):
    preds_bin = (torch.sigmoid(preds) > threshold).float()
    targets_bin = (targets > 0.5).float()

    intersection = (preds_bin * targets_bin).sum(dim=(1, 2, 3))
    union = (preds_bin + targets_bin).clamp(0, 1).sum(dim=(1, 2, 3))
    iou = (intersection + eps) / (union + eps)
    return iou.mean().item()

def test_eval_step():
    cfg = Config.fromfile('configs/carla.py')
    cfg.num_classes = 1
    cfg.batch_size = 2

    model = build_net(cfg)
    model.eval()

    dataset = build_dataset(cfg.dataset.val, cfg)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=cfg.batch_size)

    batch = next(iter(dataloader))
    imgs = batch['img']
    masks = batch['mask'].float()

    with torch.no_grad():
        logits = model(imgs)['seg']
        loss = F.binary_cross_entropy_with_logits(logits, masks)
        iou = compute_batch_iou(logits, masks)

    print(f"✅ One-batch Eval")
    print(f"Loss: {loss.item():.4f}")
    print(f"IoU:  {iou:.4f}")

if __name__ == '__main__':
    test_eval_step()
