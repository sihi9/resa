from utils import Registry, build_from_cfg

import torch
from torch.utils.data import random_split

DATASETS = Registry('datasets')

def build(cfg, registry, default_args=None):
    if isinstance(cfg, list):
        modules = [
            build_from_cfg(cfg_, registry, default_args) for cfg_ in cfg
        ]
        return nn.Sequential(*modules)
    else:
        return build_from_cfg(cfg, registry, default_args)


def _build_dataset(split_cfg, cfg):
    args = split_cfg.copy()
    args.pop('type')
    args = args.to_dict()
    args['cfg'] = cfg
    return build(split_cfg, DATASETS, default_args=args)

def build_train_val_dataloader(split_cfg, cfg):
    
    dataset = _build_dataset(split_cfg, cfg)
    
    train_split = 0.8
    seed = 42
    total_size = len(dataset)
    train_size = int(train_split * total_size)
    val_size = total_size - train_size
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)


    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size = cfg.batch_size, 
        shuffle = True,
        num_workers = cfg.workers, 
        pin_memory = False, 
        drop_last = False
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size = cfg.batch_size, 
        shuffle = False,
        num_workers = cfg.workers, 
        pin_memory = False, 
        drop_last = False
    )
     
    return train_loader, val_loader

def build_test_dataloader(split_test_cfg, cfg):

    print("Building test dataloader with config:", split_test_cfg)
    dataset = _build_dataset(split_test_cfg, cfg)

    data_loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size = cfg.batch_size, 
        shuffle = False,
        num_workers = cfg.workers, 
        pin_memory = False, 
        drop_last = False)

    return data_loader
