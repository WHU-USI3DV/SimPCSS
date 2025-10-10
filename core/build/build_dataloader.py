import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

def get_dataloader(is_source, is_train, cfg, collate_fn, dataset, shffule=True, pin_memory=True):
    if is_source and is_train:        
        loader = DataLoader(
            dataset,
            collate_fn=collate_fn,
            batch_size=cfg.src_dataloader.batch_size,
            shuffle=shffule,
            num_workers=cfg.src_dataloader.num_workers,
            pin_memory=pin_memory
        )
    elif is_source and not is_train:
        loader = DataLoader(
            dataset,
            collate_fn=collate_fn,
            batch_size=cfg.src_dataloader.batch_size * 2,
            shuffle=shffule,
            num_workers=cfg.src_dataloader.num_workers,
            pin_memory=pin_memory
        )
    elif not is_source and is_train:
        loader = DataLoader(
            dataset,
            collate_fn=collate_fn,
            batch_size=cfg.tgt_dataloader.batch_size,
            shuffle=shffule,
            num_workers=cfg.tgt_dataloader.num_workers,
            pin_memory=True
        )
    elif not is_source and not is_train:
        loader = DataLoader(
            dataset,
            collate_fn=collate_fn,
            batch_size=cfg.tgt_dataloader.batch_size * 2,
            shuffle=shffule,
            num_workers=cfg.tgt_dataloader.num_workers,
            pin_memory=True
        )

    return loader