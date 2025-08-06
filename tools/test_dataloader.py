import sys
import os
sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))

from utils.config import Config
from datasets import build_dataloader

cfg = Config.fromfile("configs/carla.py")

loader = build_dataloader(cfg.dataset.train, cfg, is_train=True)

print("✅ Dataloader loaded. Batches:", len(loader))

batch = next(iter(loader))
print("Image batch shape:", batch['img'].shape)
print("Mask batch shape:", batch['mask'].shape)
print("Mask unique values:", batch['mask'].unique())
