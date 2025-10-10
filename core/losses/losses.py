import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import yaml


class CELoss(nn.Module):
    def __init__(self, ignore_label: int = None, weight: np.ndarray = None):
        '''
        :param ignore_label: label to ignore
        :param weight: possible weights for weighted CE Loss
        '''
        super().__init__()
        if weight is not None:
            weight = torch.from_numpy(weight).float()
            print(f'----->Using weighted CE Loss weights: {weight}')

        self.loss = nn.CrossEntropyLoss(ignore_index=ignore_label, weight=weight)
        self.ignored_label = ignore_label

    def forward(self, preds: torch.Tensor, gt: torch.Tensor):

        loss = self.loss(preds, gt)
        return loss


class DICELoss(nn.Module):

    def __init__(self, ignore_label=None, powerize=True, use_tmask=True):
        super(DICELoss, self).__init__()

        if ignore_label is not None:
            self.ignore_label = torch.tensor(ignore_label)
        else:
            self.ignore_label = ignore_label

        self.powerize = powerize
        self.use_tmask = use_tmask

    def forward(self, output, target):
        input_device = output.device
        # temporal solution to avoid nan
        output = output.cpu()
        target = target.cpu()

        if self.ignore_label is not None:
            valid_idx = torch.logical_not(target == self.ignore_label)
            target = target[valid_idx]
            output = output[valid_idx, :]

        target = F.one_hot(target, num_classes=output.shape[1])
        output = F.softmax(output, dim=-1)

        intersection = (output * target).sum(dim=0)
        if self.powerize:
            union = (output.pow(2).sum(dim=0) + target.sum(dim=0)) + 1e-12
        else:
            union = (output.sum(dim=0) + target.sum(dim=0)) + 1e-12
        if self.use_tmask:
            tmask = (target.sum(dim=0) > 0).int()
        else:
            tmask = torch.ones(target.shape[1]).int()

        iou = (tmask * 2 * intersection / union).sum(dim=0) / (tmask.sum(dim=0) + 1e-12)

        dice_loss = 1 - iou.mean()

        return dice_loss.to(input_device)


def get_soft(t_vector, eps=0.25):

    max_val = 1 - eps
    min_val = eps / (t_vector.shape[-1] - 1)

    t_soft = torch.empty(t_vector.shape)
    t_soft[t_vector == 0] = min_val
    t_soft[t_vector == 1] = max_val

    return t_soft


def get_kitti_soft(t_vector, labels, eps=0.25):

    max_val = 1 - eps
    min_val = eps / (t_vector.shape[-1] - 1)

    t_soft = torch.empty(t_vector.shape)
    t_soft[t_vector == 0] = min_val
    t_soft[t_vector == 1] = max_val

    searched_idx = torch.logical_or(labels == 6, labels == 1)
    if searched_idx.sum() > 0:
        t_soft[searched_idx, 1] = max_val/2
        t_soft[searched_idx, 6] = max_val/2

    return t_soft


class SoftDICELoss(nn.Module):

    def __init__(self, ignore_label=None, powerize=True, use_tmask=True,
                 neg_range=False, eps=0.05, is_kitti=False):
        super(SoftDICELoss, self).__init__()

        if ignore_label is not None:
            self.ignore_label = torch.tensor(ignore_label)
        else:
            self.ignore_label = ignore_label
        self.powerize = powerize
        self.use_tmask = use_tmask
        self.neg_range = neg_range
        self.eps = eps
        self.is_kitti = is_kitti

    def forward(self, output, target, return_class=False, is_kitti=False):
        input_device = output.device
        # temporal solution to avoid nan
        output = output.cpu()
        target = target.cpu()

        if self.ignore_label is not None:
            valid_idx = torch.logical_not(target == self.ignore_label)
            target = target[valid_idx]
            output = output[valid_idx, :]

        target_onehot = F.one_hot(target, num_classes=output.shape[1])
        if not self.is_kitti and not is_kitti:
            target_soft = get_soft(target_onehot, eps=self.eps)
        else:
            target_soft = get_kitti_soft(target_onehot, target, eps=self.eps)

        output = F.softmax(output, dim=-1)

        intersection = (output * target_soft).sum(dim=0)

        if self.powerize:
            union = (output.pow(2).sum(dim=0) + target_soft.sum(dim=0)) + 1e-12
        else:
            union = (output.sum(dim=0) + target_soft.sum(dim=0)) + 1e-12
        if self.use_tmask:
            tmask = (target_onehot.sum(dim=0) > 0).int()
        else:
            tmask = torch.ones(target_onehot.shape[1]).int()

        iou = (tmask * 2 * intersection / union).sum(dim=0) / (tmask.sum(dim=0) + 1e-12)
        iou_class = tmask * 2 * intersection / union

        if self.neg_range:
            dice_loss = -iou.mean()
            dice_class = -iou_class
        else:
            dice_loss = 1 - iou.mean()
            dice_class = 1 - iou_class
        if return_class:
            return dice_loss.to(input_device), dice_class
        else:
            return dice_loss.to(input_device)

    
class PrototypeContrastiveLoss(nn.Module):
    def __init__(self, cfg, CL_weights: np.ndarray = None):
        super(PrototypeContrastiveLoss, self).__init__()
        self.cfg = cfg

        self.tgt2src_remap_lut_val = None
        self.src2tgt_remap_lut_val = None
        if cfg.tgt2src_mapping is not None:
            tgt2src_mapping = yaml.safe_load(open(cfg.tgt2src_mapping, 'r'))
            remap_dict_val = tgt2src_mapping["mspc2nuscenes"]
            max_key = max(remap_dict_val.keys())
            remap_lut_val = np.zeros((max_key + 1), dtype=np.int32)
            remap_lut_val[list(remap_dict_val.keys())] = list(remap_dict_val.values())
            self.tgt2src_remap_lut_val = remap_lut_val
        if cfg.src2tgt_mapping is not None:
            src2tgt_mapping = yaml.safe_load(open(cfg.src2tgt_mapping, 'r'))
            remap_dict_val = src2tgt_mapping["nuscenes2mspc"]
            max_key = max(remap_dict_val.keys())
            remap_lut_val = np.zeros((max_key + 1), dtype=np.int32)
            remap_lut_val[list(remap_dict_val.keys())] = list(remap_dict_val.values())
            self.src2tgt_remap_lut_val = torch.from_numpy(remap_lut_val).cuda()
            # self.ignore_label_list = [i for i, x in enumerate(self.src2tgt_remap_lut_val) if x == -1]
            # self.ignore_label_list.append(cfg.model.ignore_label)

        if CL_weights is not None:
            CL_weights = torch.from_numpy(CL_weights).float()
            print(f'----->Using weighted CL Loss weights: {CL_weights}')
        self.ce_criterion = nn.CrossEntropyLoss(ignore_index=cfg.model.ignore_label, weight=CL_weights)

    def forward(self, Proto, feat, labels):
        """
        Args:
            C: NUM_CLASSES A: feat_dim B: batch_size H: feat_high W: feat_width N: number of pixels except IGNORE_LABEL
            Proto: shape: (C, A) the mean representation of each class
            feat: shape (BHW, A) -> (N, A)
            labels: shape (BHW, ) -> (N, )
        Returns:
        """
        # assert not Proto.requires_grad
        assert not labels.requires_grad
        assert feat.requires_grad
        assert feat.dim() == 2
        assert labels.dim() == 1
        # remove IGNORE_LABEL pixels
        # if self.src2tgt_remap_lut_val is not None:
        #     mask = [item not in self.ignore_label_list for item in labels]
        # else:
        #     mask = (labels != self.cfg.model.ignore_label)

        if self.src2tgt_remap_lut_val is not None:
            labels = self.src2tgt_remap_lut_val[labels].long()

        mask = (labels != self.cfg.model.ignore_label)
        labels = labels[mask]
        feat = feat[mask]

        feat = F.normalize(feat, p=2, dim=1)

        if self.tgt2src_remap_lut_val is not None:
            self.mapped_Proto = torch.zeros_like(Proto)
            for i in range(len(self.tgt2src_remap_lut_val)):
                if self.tgt2src_remap_lut_val[i] != -1:
                    self.mapped_Proto[i, :] = Proto[i, :]

        Proto = F.normalize(self.mapped_Proto if self.tgt2src_remap_lut_val is not None else Proto, p=2, dim=1)

        # Proto = F.normalize(Proto, p=2, dim=1)

        logits = feat.mm(Proto.permute(1, 0).contiguous())
        
        logits = logits / self.cfg.distill.tau
    
        loss = self.ce_criterion(logits, labels)
        
        return loss