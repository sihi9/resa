import torch
from tqdm import tqdm
from torch.nn.functional import sigmoid
from runner.registry import EVALUATOR
from datasets.registry import build_dataset

@EVALUATOR.register_module
class BinaryIoUEvaluator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.threshold = 0.5
        self.loss_fn = torch.nn.BCEWithLogitsLoss()

        self.dataset = build_dataset(cfg.dataset.val, cfg)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.workers
        )

    def compute_batch_iou(self, preds, targets, eps=1e-6):
        preds_bin = (sigmoid(preds) > self.threshold).float()
        targets_bin = (targets > 0.5).float()

        intersection = (preds_bin * targets_bin).sum(dim=(1, 2, 3))
        union = (preds_bin + targets_bin).clamp(0, 1).sum(dim=(1, 2, 3))
        iou = (intersection + eps) / (union + eps)
        return iou.mean().item()

    def __call__(self, net):
        net.eval()
        net.to(self.device)

        total_iou = 0.0
        total_loss = 0.0
        total_batches = 0

        with torch.no_grad():
            for batch in tqdm(self.dataloader, desc='[Carla Eval]'):
                imgs = batch['img'].to(self.device)
                masks = batch['mask'].float().to(self.device)

                logits = net(imgs)['seg']  # [B, 1, H, W]

                loss = self.loss_fn(logits, masks)
                iou = self.compute_batch_iou(logits, masks)

                total_loss += loss.item()
                total_iou += iou
                total_batches += 1

        print(f"\n✅ Carla Evaluation Complete")
        print(f"Avg BCE Loss: {total_loss / total_batches:.4f}")
        print(f"Avg IoU:      {total_iou / total_batches:.4f}")
