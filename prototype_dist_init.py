import argparse
import logging
import numpy as np
from collections import OrderedDict
from tqdm import tqdm
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch.distributed
import torch.backends.cudnn
from torch.utils.data import DataLoader
from torchvision.models._utils import IntermediateLayerGetter

import core.models as models
import MinkowskiEngine as ME
from core.datasets.build import get_dataset
from core.models.collation import CollateFN

from configs.config import get_config
from core.utils.metric import mkdir, AverageMeter, intersectionAndUnionGPU, load_model
from core.utils.logger import setup_logger
from core.utils.prototype_estimator import prototype_estimator

import warnings
warnings.filterwarnings('ignore')


def prototype_dist_init(cfg, logger):
    # load init source prototypes
    logger.info(">>>>>>>>>>>>>>>> Load init prototypes >>>>>>>>>>>>>>>>")
    
    model = getattr(models, cfg.model.name)

    out_feature_name = 96
    # 选取参与的block，但没有hb_learner
    RETURN_BLOCKS = ['block4', 'block5', 'block6', 'block7', 'block8']
    feature_extractor = model(cfg.model.in_feat_size, cfg.model.out_classes, 3, True, RETURN_BLOCKS=RETURN_BLOCKS)
    classifier = models.classifier(out_feature_name, cfg.model.out_classes, 3)
    
    device = torch.device(cfg.pipeline.gpus)
    feature_extractor.to(device)
    classifier.to(device)

    # load init source prototypes
    estimator = {}
    # 哪些层参与特征增强
    CL_LAYERS = cfg.distill.CL_BLOCKS
    for k in CL_LAYERS:
        estimator.update({k: prototype_estimator(k, cfg=cfg)})

    if cfg.distill.find_TP:
        stride_mapping = {'block4': 16, 'block5': 8, 'block6': 4, 'block7': 2}
        logits_pool_dict = nn.ModuleDict()
        for k, v in estimator.items():
            if k == 'block8':
                continue
            logits_pool = ME.MinkowskiAvgPooling(kernel_size=stride_mapping[k], stride=stride_mapping[k], dimension=3)
            logits_pool_dict.update({k: logits_pool})
        logits_pool_dict.to(device)
    
    if cfg.pipeline.resume_hb:
        logger.info("Loading checkpoint from {}".format(cfg.pipeline.resume_hb))
        feature_extractor = load_model(cfg.pipeline.resume_hb, feature_extractor, False)
        classifier = load_model(cfg.pipeline.resume_hb, classifier, False)

    torch.cuda.empty_cache()

    tgt_train_data = get_dataset(cfg, mode='train', is_source=False)

    collation = CollateFN()

    train_loader = DataLoader(
        tgt_train_data,
        collate_fn = collation,
        batch_size = cfg.tgt_dataloader.batch_size,
        shuffle = False,
        num_workers = cfg.tgt_dataloader.num_workers,
        pin_memory = True
    )

    def get_pure_downsampled_labels(cfg, labels, coords):
        labels_down_dict = {}
        pure_indices_dict = {}
        # +1是为了将-1的ignore_label变为0,否则无法进行池化
        labels_onehot = F.one_hot(labels + 1, num_classes = cfg.model.out_classes + 1)      
        labels_onehot_tensor = ME.SparseTensor(features=labels_onehot.to(dtype=torch.float), coordinates=coords)
        labels_onehot_tensor.detach()
        for k in RETURN_BLOCKS:
            if k == 'block8':
                continue
            labels_logits_stensor = logits_pool_dict[k](labels_onehot_tensor)
            # 在较低分辨率下,确保一个格网内,某一个类的数量大于一定阈值时才保留
            vox_max_values, vox_max_indices = torch.max(labels_logits_stensor.F, dim=1)
            vox_keep_indices = (vox_max_values >= torch.tensor(cfg.distill.DELTA))
            pure_indices_dict.update({k: vox_keep_indices})
            labels_down = vox_max_indices -1
            labels_down_dict.update({k: labels_down})
        return pure_indices_dict, labels_down_dict # 注意,labels_down不是pure的!

    # prototype initialize
    feature_extractor.eval()
    classifier.eval()
    logits_pool_dict.eval()

    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()
    # pbar for presenting information in real time
    train_pbar = tqdm(train_loader, total=len(train_loader))
    with torch.no_grad():
        for i, data in enumerate(train_pbar):
            coords = data['coordinates'].int().to(device)
            feats = data['features'].float().to(device)
            labels = data['labels'].long().to(device)
            
            pure_indices_dict, labels_down_dict = get_pure_downsampled_labels(cfg=cfg, labels=labels, coords=coords)

            stensor = ME.SparseTensor(features=feats, coordinates=coords)
            out_feat, block_feat = feature_extractor(stensor)
            src_out = classifier(out_feat)
            pred = src_out.F.max(dim=-1).indices
            if cfg.distill.find_TP:
                for k in CL_LAYERS:
                    if k == 'block8':
                        keep_idx = (pred==labels)
                        estimator[k].update(features=block_feat[k].F.detach()[keep_idx], labels=labels[keep_idx])
                        continue
                    _, vox_pred = get_pure_downsampled_labels(cfg=cfg, labels=pred, coords=coords)
                    keep_idx = (vox_pred[k]==labels_down_dict[k]) & pure_indices_dict[k]
                    estimator[k].update(features=block_feat[k].F.detach()[keep_idx], labels=labels_down_dict[k][keep_idx])
            else:
                for k in CL_LAYERS:
                    if k == 'block8':
                        estimator[k].update(features=block_feat[k].F.detach(), labels=labels)
                        continue
                    estimator[k].update(features=block_feat[k].F.detach(), labels=labels_down_dict[k])

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

        logger.info('Val results: mIoU {:.4f}'.format(mIoU))
        for i in range(cfg.model.out_classes):
            logger.info('class_{} Result: iou {:.4f}.'.format(i, iou_class[i]))

    out_dir = cfg.distill.CV_DIR

    for k in CL_LAYERS:
        estimator[k].save(name=os.path.join(out_dir, 'prototype_feat_dist_' + k + '.pth'))


def main():
    parser = argparse.ArgumentParser(description="PyTorch Semantic Segmentation Training")
    parser.add_argument(
        "-cfg",
        "--config_file",
        default="",
        type=str,
        help="Path to config file"
    )

    args = parser.parse_args()
    cfg = get_config(args.config_file)

    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        mkdir(output_dir)

    logger = setup_logger("prototype_dist_init", output_dir)
    logger.info("Using {} GPUS.".format(cfg.pipeline.gpus))
    logger.info(args)

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    prototype_dist_init(cfg, logger)

if __name__ == "__main__":
    main()