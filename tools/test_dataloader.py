import sys
import os
sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))

import torch
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F

from utils.config import Config
from datasets import build_train_val_dataloader, build_test_dataloader

# --- Load Config & Dataloader ---
cfg = Config.fromfile("configs/carla.py")
train_loader, val_loader = build_train_val_dataloader(cfg.dataset.train, cfg)
test_loader = build_test_dataloader(cfg.dataset.test, cfg)
print("✅ Dataloader loaded. Batches:", len(test_loader))

batch = next(iter(val_loader))
imgs = batch['img']          # [B, 3, H, W]
masks = batch['mask']        # [B, 1, H, W]

print("Image batch shape:", imgs.shape)
print("Mask batch shape:", masks.shape)
print("Mask unique values:", masks.unique())

# --- Choose sample ---
img = imgs[0]                # Tensor: [3, H, W]
mask = masks[0][0]           # Tensor: [H, W]

# --- Denormalize Image ---
mean = torch.tensor(cfg.img_norm['mean']).view(3, 1, 1)
std = torch.tensor(cfg.img_norm['std']).view(3, 1, 1)
img_denorm = img * std + mean

# --- If using ImageNet mean/std, clamp to [0,1] for safe display ---
if max(cfg.img_norm['mean']) <= 1.0:
    img_denorm = img_denorm.clamp(0, 1)

# --- Show Image + Mask ---
fig, axs = plt.subplots(1, 2, figsize=(10, 4))
axs[0].imshow(img_denorm.permute(1, 2, 0).cpu().numpy())
axs[0].set_title("Denormalized Image")
axs[0].axis('off')

axs[1].imshow(mask.cpu().numpy(), cmap='gray')
axs[1].set_title("Binary Mask")
axs[1].axis('off')
plt.tight_layout()
plt.show()

# --- Print Image Stats ---
print("Normalized Image Mean:", img.mean(dim=(1, 2)))
print("Normalized Image Std:", img.std(dim=(1, 2)))

print("Denormalized Image Mean:", img_denorm.mean(dim=(1, 2)))
print("Denormalized Image Std:", img_denorm.std(dim=(1, 2)))
