import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import torch
from utils.config import Config
from models.registry import build_net

def test_forward():
    # Load Carla config
    cfg = Config.fromfile('configs/carla.py')  # Make sure this path is correct

    # Force num_classes = 1 to be safe
    cfg.num_classes = 1
    cfg.batch_size = 1
    cfg.img_height = 360
    cfg.img_width = 640

    # Build model
    model = build_net(cfg)
    model.eval()

    # Create dummy input tensor
    dummy_input = torch.randn(cfg.batch_size, 3, cfg.img_height, cfg.img_width)

    # Forward pass
    with torch.no_grad():
        output = model(dummy_input)

    seg = output['seg']

    print("Output shape:", seg.shape)
    print("Output min/max:", seg.min().item(), seg.max().item())

    # Assertions
    assert seg.shape == (cfg.batch_size, 1, cfg.img_height, cfg.img_width), "Incorrect output shape"
    print("✅ Test passed!")

if __name__ == '__main__':
    test_forward()
