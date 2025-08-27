# datasets/carla.py
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from .registry import DATASETS

@DATASETS.register_module
class CarlaLaneDataset(Dataset):
    def __init__(self, root, folder_name='Town04_2000', img_size=(360, 640), transform=None, cfg=None):
        self.cfg = cfg
        self.root = root
        self.img_size = img_size
        self.transform = transform

        # Accept either a string or a list of folder names
        if isinstance(folder_name, str):
            folder_names = [folder_name]
        else:
            folder_names = folder_name

        self.img_paths = []
        self.mask_paths = []
        for name in folder_names:
            folder_root = os.path.join(self.root, name)
            rgb_dir = os.path.join(folder_root, 'rgb_downscaled',)
            label_dir = os.path.join(folder_root, 'labels_downscaled')

            img_paths = sorted([
                os.path.join(rgb_dir, fname)
                for fname in os.listdir(rgb_dir)
                if fname.endswith('.jpg')
            ])
            mask_paths = [
                os.path.join(label_dir, os.path.splitext(os.path.basename(p))[0] + '.png')
                for p in img_paths
            ]

            self.img_paths.extend(img_paths)
            self.mask_paths.extend(mask_paths)

        self.to_tensor = T.ToTensor()
        self.resize = T.Resize(img_size, interpolation=Image.BILINEAR)
        self.mask_resize = T.Resize(img_size, interpolation=Image.NEAREST)

        self.normalize = T.Normalize(
            mean=self.cfg.img_norm['mean'],
            std=self.cfg.img_norm['std']
        )

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert('RGB')
        mask = Image.open(self.mask_paths[idx]).convert('L')

        img = self.resize(img)
        mask = self.mask_resize(mask)

        img = self.to_tensor(img)
        img = self.normalize(img)

        mask = self.to_tensor(mask)
        mask = (mask > 0.5).float()

        return {'img': img, 'mask': mask}
