from json import load
import os
from pickletools import optimize
from sre_parse import State
import time
import argparse
import datetime
import logging
import numpy as np
import errno
from tqdm import tqdm
from collections import OrderedDict
from sklearn.metrics import jaccard_score

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import MinkowskiEngine as ME

from configs.config import get_config
from core.losses import CELoss, DICELoss, SoftDICELoss
from core.utils.metric import mkdir, AverageMeter, intersectionAndUnionGPU, load_model
from core.datasets.build import get_dataset
from core.utils.logger import setup_logger
from core.utils.metric_logger import MetricLogger
from core.models.collation import CollateFN
import core.models as models
import core.build as build


class Trainer:
    def __init__(self, cfg, args):
        self.logger = logging.getLogger("PC_DA.Trainer")
        self.logger.info("Start training")
        self.validate_every_epoch = args.test

        model = getattr(models, cfg.model.name)
        self.model = model(cfg.model.in_feat_size, cfg.model.out_classes)

        # Single GPU
        self.device = torch.device('cuda')
        self.model.to(self.device)

        self.optimizer = build.get_optimizer(cfg, self.model)
        self.optimizer.zero_grad()

        self.scheduler = build.get_scheduler(cfg, self.optimizer)

        if cfg.pipeline.resume:
            self.logger.info("Loading checkpoint from {}".format(cfg.pipeline.resume))
            self.model = load_model(cfg.pipeline.resume, self.model, False)

        self.src_train_data = get_dataset(cfg, mode='train', is_source=True)
        self.src_val_data = get_dataset(cfg, mode='val', is_source=True)

        if cfg.pipeline.criterion == 'SoftDICELoss':
            if cfg.model.out_classes == 19:
                self.criterion = SoftDICELoss(ignore_label=cfg.model.ignore_label, is_kitti=True).to(self.device)
            else:
                self.criterion = SoftDICELoss(ignore_label=cfg.model.ignore_label).to(self.device)
        elif cfg.pipeline.criterion == 'DICELoss':
            self.criterion = DICELoss(ignore_label=cfg.model.ignore_label)
        elif cfg.pipeline.criterion == 'CELoss':
            if cfg.source_dataset.balance_weights:
                self.weight = self.src_train_data.get_dataset_weights()
                self.criterion = CELoss(ignore_label=cfg.model.ignore_label, weight=self.weight).to(self.device)
            else:
                self.criterion = CELoss(ignore_label=cfg.model.ignore_label).to(self.device)

        collation = CollateFN()

        self.train_loader = DataLoader(
            self.src_train_data,
            collate_fn=collation,
            batch_size=cfg.dataloader.batch_size,
            shuffle=True,
            num_workers=cfg.dataloader.num_workers,
            pin_memory=True
        )

        self.val_loader = DataLoader(
            self.src_val_data,
            collate_fn=collation,
            batch_size=cfg.dataloader.batch_size,
            shuffle=False,
            num_workers=cfg.dataloader.num_workers,
            pin_memory=True
        )

    def train(self, cfg):
        max_mIoU, tot_iter, max_mIoU_epoch = 0, 0, 0
        meters = MetricLogger(delimiter='   ')
        for epoch in range(cfg.pipeline.epochs):
            # pbar for presenting information in real time
            train_pbar = tqdm(total=len(self.train_loader))

            torch.cuda.empty_cache()
            self.model.train()
            for i, data in enumerate(self.train_loader):
                coords = data['coordinates'].int().to(self.device)
                feats = data['features'].float().to(self.device)
                labels = data['labels'].long().to(self.device)
                stensor = ME.SparseTensor(features=feats, coordinates=coords)
                pred = self.model(stensor)
                loss = self.criterion(pred.F, labels)
                meters.update(loss=loss.item())

                loss.backward()

                self.optimizer.step()
                self.optimizer.zero_grad()

                tot_iter += 1
                train_pbar.set_postfix(loss=float(loss.detach().cpu().numpy()),
                                       lr=self.optimizer.state_dict()['param_groups'][0]['lr'])
                train_pbar.update(1)

                # MinkowskiEngine needs to clear the cache at regular interval.
                # For the reason, please check at the official document.
                torch.cuda.empty_cache()

                if tot_iter % 100 == 0 or tot_iter == 1:
                    self.logger.info("Epoch: {epoch} iter: {iter} {meters} lr: {lr}".format
                        (
                        epoch=epoch,
                        iter=tot_iter,
                        meters=str(meters),
                        lr=self.optimizer.state_dict()['param_groups'][0]['lr']
                    )
                    )

            # update the LR
            if cfg.pipeline.scheduler is not None:
                self.scheduler.step()

            # test for every epoch
            if self.validate_every_epoch:
                mIoU = self.validate(cfg)
            # save parameters if the mIoU becomes the best 
                if mIoU > max_mIoU:
                    max_mIoU = mIoU
                    max_mIoU_epoch = epoch
                    filename = os.path.join(cfg.OUTPUT_DIR, "model_epoch{:03d}.pth".format(epoch))
                    torch.save({'epoch': epoch, 'model': self.model.state_dict()}, filename)

        if not self.validate_every_epoch:
            mIoU = self.validate(cfg)
            max_mIoU = mIoU
            max_mIoU_epoch = epoch
            filename = os.path.join(cfg.OUTPUT_DIR, "model_epoch{:03d}.pth".format(epoch))
            torch.save({'epoch': epoch, 'model': self.model.state_dict()}, filename)
        self.logger.info('Best result is got at epoch {} with mIoU {:.4f}.'.format(max_mIoU_epoch, max_mIoU))

    def validate(self, cfg):
        self.logger = logging.getLogger("PC_DA.tester")

        self.model.eval()

        torch.cuda.empty_cache()

        intersection_meter = AverageMeter()
        union_meter = AverageMeter()
        target_meter = AverageMeter()

        # pbar for presenting information in real time
        val_pbar = tqdm(self.val_loader, total=len(self.val_loader))

        with torch.no_grad():
            for i, data in enumerate(val_pbar):
                coords = data['coordinates'].int().to(self.device)
                feats = data['features'].float().to(self.device)
                labels = data['labels'].long().to(self.device)
                stensor = ME.SparseTensor(features=feats, coordinates=coords)

                out = self.model(stensor).F
                pred = out.max(dim=-1).indices

                intersection, union, target = intersectionAndUnionGPU(
                    pred, labels, cfg.model.out_classes, cfg.model.ignore_label
                )

                intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
                intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)

                if i % 2 == 0:
                    # MinkowskiEngine needs to clear the cache at regular interval.
                    # For the reason, please check at the official document.
                    torch.cuda.empty_cache()

            iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
            mIoU = np.mean(iou_class)

            self.logger.info('Val results: mIoU {:.4f}'.format(mIoU))
            for i in range(cfg.model.out_classes):
                self.logger.info('class_{} Result: iou {:.4f}.'.format(i, iou_class[i]))

            return mIoU


def main():
    parser = argparse.ArgumentParser(description="Source domain training")
    parser.add_argument(
        "-cfg",
        "--config_file",
        default="",
        type=str,
        help="Path to config file"
    )
    parser.add_argument('-test', action='store_false', default=True, help='test for every epoch')

    args = parser.parse_args()
    cfg = get_config(args.config_file)

    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        mkdir(output_dir)

    logger = setup_logger("PC_DA", output_dir)
    logger.info("Using {} GPUS.".format(cfg.pipeline.gpus))
    logger.info(args)

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    # fix random seed
    os.environ['PYTHONHASHSEED'] = str(cfg.pipeline.seed)
    np.random.seed(cfg.pipeline.seed)
    torch.manual_seed(cfg.pipeline.seed)
    torch.cuda.manual_seed(cfg.pipeline.seed)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    trainer = Trainer(cfg, args)
    trainer.train(cfg)


if __name__ == '__main__':
    main()
