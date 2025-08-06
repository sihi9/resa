import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

from datasets.carla import CarlaLaneDataset

dataset = CarlaLaneDataset(root='data/Carla', folder_name='Town04_2000')
sample = dataset[0]

print("Image shape:", sample['img'].shape)   # [3, 360, 640]
print("Mask shape:", sample['mask'].shape)   # [1, 360, 640]
print("Mask values:", sample['mask'].unique())  # expect: tensor([0., 1.])
