from PIL import Image
import os
import torch
from torchvision import transforms
from tqdm import tqdm

img_dir = 'data/Carla/Town06_5000/rgb_downscaled'
transform = transforms.ToTensor()

mean_sum = torch.zeros(3)
std_sum = torch.zeros(3)
count = 0

for fname in tqdm(os.listdir(img_dir)):
    if not fname.endswith('.jpg'):
        continue
    img = Image.open(os.path.join(img_dir, fname)).convert('RGB')
    tensor = transform(img)  # [C, H, W]
    mean_sum += tensor.mean(dim=(1, 2))
    std_sum += tensor.std(dim=(1, 2))
    count += 1

mean = mean_sum / count
std = std_sum / count

print("Dataset Mean:", mean)
print("Dataset Std:", std)
