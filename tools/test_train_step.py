import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from utils.config import Config
from models.registry import build_net
from runner.registry import build as build_trainer
from runner import registry as trainer_registry

def test_train_step():
    # Load Carla config
    cfg = Config.fromfile('configs/carla.py')

    # Force single-channel binary settings
    cfg.num_classes = 1
    cfg.seg_loss_weight = 1.0
    cfg.loss_type = 'bce'  # For clarity, even though we override loss anyway
    cfg.img_height = 360
    cfg.img_width = 640
    cfg.batch_size = 2

    # Dummy batch
    B, C, H, W = cfg.batch_size, 3, cfg.img_height, cfg.img_width
    dummy_img = torch.randn(B, C, H, W)
    dummy_mask = torch.randint(0, 2, (B, H, W)).float()  # binary {0.0, 1.0}

    # Build model and trainer
    model = build_net(cfg)
    trainer = build_trainer(cfg.trainer, registry=trainer_registry.TRAINER, default_args=dict(cfg=cfg))



    # Forward pass through trainer
    model.eval()
    with torch.no_grad():
        batch = {'img': dummy_img, 'label': dummy_mask}
        out = trainer(model, batch)

    print("Loss:", out['loss'].item())
    for k, v in out['loss_stats'].items():
        print(f"{k}: {v.item()}")

    # Assertions
    assert torch.is_tensor(out['loss'])
    assert out['loss'].item() > 0
    print("✅ Train step test passed!")

if __name__ == '__main__':
    test_train_step()
