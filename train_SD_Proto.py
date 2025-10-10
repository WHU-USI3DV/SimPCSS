import os
from pickletools import optimize
from sre_parse import State
import argparse
import logging
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import MinkowskiEngine as ME

from configs.config import get_config
from core.losses import CELoss, DICELoss, SoftDICELoss, PrototypeContrastiveLoss
from core.utils.metric import mkdir, AverageMeter, intersectionAndUnionGPU, load_model
from core.datasets.build import get_dataset
from core.utils.logger import setup_logger
from core.utils.metric_logger import MetricLogger
from core.models.collation import CollateFN
import core.models as models
import torch.distributed as dist
from core.utils.prototype_estimator import prototype_estimator
import core.build as build


class Trainer:
    def __init__(self, cfg, args):
        self.logger = logging.getLogger("PC_DA.Trainer")
        self.logger.info("Start training")
        self.validate_every_epoch = args.test

        # 哪些层参与特征增强，其特征都经过hb_learner
        self.CL_LAYERS = cfg.distill.CL_BLOCKS
        # 选取参与的block，但没有hb_learner
        self.RETURN_BLOCKS = cfg.distill.RETURN_BLOCKS

        model = getattr(models, cfg.model.name)
        # 两个不同线数的训练的backbone不同 model_lb以及model_hb分别对应low beam和high beam
        self.model_lb = model(cfg.model.in_feat_size, cfg.model.out_classes, 3, True, CL_LAYERS=self.CL_LAYERS)
        self.classifier_lb = models.classifier(96, cfg.model.out_classes, 3)

        if cfg.pipeline.resume_lb:
            self.logger.info("Loading checkpoint from {}".format(cfg.pipeline.resume_lb))
            self.model_lb = load_model(cfg.pipeline.resume_lb, self.model_lb, True)
            self.classifier_lb = load_model(cfg.pipeline.resume_lb, self.classifier_lb, True)

        self.device = torch.device(cfg.pipeline.gpus)
        self.model_lb.to(self.device)
        self.classifier_lb.to(self.device)

        # load init source prototypes
        self.logger.info(">>>>>>>>>>>>>>>> Load init prototypes >>>>>>>>>>>>>>>>")
        self.estimator = {}
        for k in self.CL_LAYERS:
            self.estimator.update({k: prototype_estimator(k, cfg=cfg)})

        # 如果需要找到各个level的tp，可以使用AvgPool实现高效获取
        if cfg.distill.find_TP:
            stride_mapping = {'block4': 16, 'block5': 8, 'block6': 4, 'block7': 2}
            self.logits_pool_dict = nn.ModuleDict()
            for k in self.CL_LAYERS:
                if k == 'block8':
                    continue
                logits_pool = ME.MinkowskiAvgPooling(kernel_size=stride_mapping[k], stride=stride_mapping[k], dimension=3)
                self.logits_pool_dict.update({k: logits_pool})
            self.logits_pool_dict.to(self.device)

        torch.autograd.set_detect_anomaly(True)
        
        self.optimizer_lb = build.get_optimizer(cfg, self.model_lb, self.classifier_lb)
        self.scheduler_lb = build.get_scheduler(cfg, self.optimizer_lb)

        self.src_train_data = get_dataset(cfg, mode='train', is_source=True, return_blocks=self.RETURN_BLOCKS)
        self.val_data = get_dataset(cfg, mode='val', is_source=True)

        collation = CollateFN()
        # def get_dataloader(is_source, is_train, cfg, collate_fn, dataset, shffule=True, pin_memory=True):
        self.src_train_loader =  build.get_dataloader(True, True, cfg, collation, self.src_train_data)
        self.val_loader =  build.get_dataloader(True, False, cfg, collation, self.val_data)

        # Init contrastiveLoss for KD
        if cfg.distill.balance_weights:
            self.src_weight = self.src_train_data.get_dataset_weights()
            self.pcl_criterion = PrototypeContrastiveLoss(cfg, CL_weights=self.src_weight).to(self.device)
        else:      
            self.pcl_criterion = PrototypeContrastiveLoss(cfg).to(self.device)

        # Init crossentopyLoss for seg
        if cfg.source_dataset.balance_weights:
            self.src_weight = self.src_train_data.get_dataset_weights()
            self.src_criterion = CELoss(ignore_label=cfg.model.ignore_label, weight=self.src_weight).to(self.device)
        else:
            self.src_criterion = CELoss(ignore_label=cfg.model.ignore_label).to(self.device)


    def get_pure_downsampled_labels(self, cfg, labels, coords):
        labels_down_dict = {}
        pure_indices_dict = {}
        # +1是为了将-1的ignore_label变为0,否则无法进行池化
        labels_onehot = F.one_hot(labels + 1, num_classes = cfg.model.out_classes + 1)      
        labels_onehot_tensor = ME.SparseTensor(features=labels_onehot.to(dtype=torch.float), coordinates=coords)
        labels_onehot_tensor.detach()
        for k in self.CL_LAYERS:
            if k == 'block8':
                continue
            labels_logits_stensor = self.logits_pool_dict[k](labels_onehot_tensor)
            # 在较低分辨率下,确保一个格网内,某一个类的数量大于一定阈值时才保留
            vox_max_values, vox_max_indices = torch.max(labels_logits_stensor.F, dim=1)
            vox_keep_indices = (vox_max_values >= torch.tensor(cfg.distill.DELTA))
            pure_indices_dict.update({k: vox_keep_indices})
            labels_down = vox_max_indices -1
            labels_down_dict.update({k: labels_down})
        return pure_indices_dict, labels_down_dict # 注意,labels_down不是pure的!


    def train(self, cfg):
        max_mIoU, max_mIoU_epoch, tot_iter = 0, 0, 0
        meters = MetricLogger(delimiter='   ')
        for epoch in range(cfg.pipeline.epochs):
            # pbar for presenting information in real time
            train_pbar = tqdm(total=len(self.src_train_loader))

            self.model_lb.train()
            self.classifier_lb.train()
            for i, src_data in enumerate(self.src_train_loader):
                src_coords = src_data['coordinates'].int().to(self.device)
                src_feats = src_data['features'].float().to(self.device)
                src_labels = src_data['labels'].long().to(self.device)
                src_stensor = ME.SparseTensor(features=src_feats, coordinates=src_coords)
                src_out_feat, hb_learner_feature = self.model_lb(src_stensor)
                src_out = self.classifier_lb(src_out_feat)

                # 两个域各自的分割损失
                loss_seg_src = self.src_criterion(src_out.F, src_labels)

                loss = torch.tensor(0.).cuda()
                loss += loss_seg_src

                meters.update(loss_seg_src=loss_seg_src.item())

                if cfg.distill.find_TP:
                    src_labels_down_indices, src_labels_down = self.get_pure_downsampled_labels(cfg=cfg, labels=src_labels, coords=src_coords)

                # 让增强特征学习器得到的特征和类原型接近
                for k in self.CL_LAYERS:
                    # hb_learner_feature[k].detach()
                    if k == 'block8':          
                        loss_CL = self.pcl_criterion(Proto=self.estimator[k].Proto.detach(),
                                                feat=self.model_lb.hb_learner[k](hb_learner_feature[k].detach()).F,
                                                labels=src_labels)
                        loss += cfg.distill.LAMBDA_FEAT * loss_CL
                        meters.update(loss_CL_block8=loss_CL.item())
                        continue
                    loss_CL = self.pcl_criterion(Proto=self.estimator[k].Proto.detach(),
                                                feat=self.model_lb.hb_learner[k](hb_learner_feature[k].detach()).F,
                                                labels=src_labels_down[k])
                    # loss_CL_tot += loss_CL
                    loss += cfg.distill.LAMBDA_FEAT * loss_CL
                    if k == 'block7':
                        meters.update(loss_CL_block7=loss_CL.item())
                    elif k == 'block6':
                        meters.update(loss_CL_block6=loss_CL.item())
                    elif k == 'block5':
                        meters.update(loss_CL_block5=loss_CL.item())
                    elif k == 'block4':
                        meters.update(loss_CL_block4=loss_CL.item())
                    
                # 筛选出source标签中置信度大于阈值且预测正确的标签
                # confidence_th = torch.tensor(0.8).cuda()
                # target_pseudo = F.softmax(tgt_out.F, dim=1)
                # target_conf, target_pseudo = target_pseudo.max(dim = 1)
                # selected_indics = torch.nonzero((target_conf > confidence_th) & (target_pseudo.long() == tgt_labels))
                # valid_idx = target_conf > confidence_th & target_pseudo == tgt_labels
                # filtered_target_pseudo = - torch.ones_like(target_pseudo)
                # valid_idx = target_conf > confidence_th
                # filtered_target_pseudo[valid_idx] = target_pseudo[valid_idx]
                # target_pseudo = filtered_target_pseudo.long()

                loss.backward()

                self.optimizer_lb.step()
                self.optimizer_lb.zero_grad()

                # MinkowskiEngine needs to clear the cache at regular interval.
                # For the reason, please check at the official document.
                torch.cuda.empty_cache()

                tot_iter += 1
                train_pbar.set_postfix(loss_seg_src=float(loss_seg_src.detach().cpu().numpy()),
                                       lr=self.optimizer_lb.state_dict()['param_groups'][0]['lr'])
                train_pbar.update(1)

                if tot_iter % 100 == 0 or tot_iter == 1:
                    self.logger.info("Epoch: {epoch} iter: {iter} {meters} lr: {lr}".format
                        (
                        epoch=epoch,
                        iter=tot_iter,
                        meters=str(meters),
                        lr=self.optimizer_lb.state_dict()['param_groups'][0]['lr']
                    )
                    )

            # update the LR
            if cfg.pipeline.scheduler is not None:
                self.scheduler_lb.step()

            # test for every epoch
            if self.validate_every_epoch:
                mIoU = self.validate(cfg)
                # save parameters if the mIoU becomes the best 
                if mIoU > max_mIoU:
                    max_mIoU = mIoU
                    max_mIoU_epoch = epoch
                    filename = os.path.join(cfg.OUTPUT_DIR, "model_epoch{:03d}.pth".format(epoch))
                    torch.save({'epoch': epoch, 'model': self.model_lb.state_dict(),
                                'classifier': self.classifier_lb.state_dict(), 
                                }, filename)
        if not self.validate_every_epoch:
            mIoU = self.validate(cfg)
            max_mIoU = mIoU
            max_mIoU_epoch = epoch
            filename = os.path.join(cfg.OUTPUT_DIR, "model_epoch{:03d}.pth".format(epoch))
            torch.save({'epoch': epoch, 'model': self.model_lb.state_dict(),
                        'classifier': self.classifier_lb.state_dict(), 
                        #'hb_learner': self.HB_learner_out.state_dict(), 
                        }, filename)
        self.logger.info('Best result is got at epoch {} with mIoU {:.4f}.'.format(max_mIoU_epoch, max_mIoU))


    def validate(self, cfg):
        self.logger = logging.getLogger("PC_DA.tester")

        self.model_lb.eval()
        self.classifier_lb.eval()

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
                
                src_feat, _ = self.model_lb(stensor)
                out = self.classifier_lb(src_feat).F
                pred = out.max(dim=-1).indices

                intersection, union, target = intersectionAndUnionGPU(
                    pred, labels, cfg.model.out_classes, cfg.model.ignore_label
                )

                intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
                intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)

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
    parser.add_argument('--test', action='store_false', default=True, help='test for every epoch')

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
