import os
import torch
import torch.utils.data
import torch.distributed
import torch.backends.cudnn
import torch.nn.functional as F


class prototype_estimator():
    def __init__(self, feature_name, cfg):
        super(prototype_estimator, self).__init__()

        self.cfg = cfg
        self.class_num = cfg.model.out_classes
        # _, backbone_name = cfg.MODEL.NAME.split('_')
        self.num_dict = {'block4':256, 'block5':256, 'block6':128, 'block7':96, 'block8':96}
        # self.out_feature_num = 96
        # self.bottle_feature_num = 256

        # momentum 
        self.use_momentum = cfg.distill.use_momentum
        self.momentum = cfg.distill.momentum

        # init prototype
        self.init(feature_name=feature_name, resume=self.cfg.distill.CV_DIR)

    def init(self, feature_name, resume=""):
        if resume:
            path = 'prototype_feat_dist_' + feature_name + '.pth'
            print(path)
            resume = os.path.join(resume, path)
            if not os.path.exists(resume):
                raise RuntimeError("pth not available: {}".format(resume))
            print("Loading checkpoint from {}".format(resume))
            checkpoint = torch.load(resume, map_location=torch.device('cpu'))
            self.Proto = checkpoint['Proto'].cuda(non_blocking=True)
            self.Amount = checkpoint['Amount'].cuda(non_blocking=True)
        else:
            self.Proto = torch.zeros(self.class_num, self.num_dict[feature_name]).cuda(non_blocking=True)
            self.Amount = torch.zeros(self.class_num).cuda(non_blocking=True)

    def update(self, features, labels):
        A=self.Proto.detach().clone()
        mask = (labels != self.cfg.model.ignore_label)
        # remove IGNORE_LABEL pixels
        labels = labels[mask]
        features = features[mask]
        if not self.use_momentum:
            N, A = features.size()
            C = self.class_num
            # refer to SDCA for fast implementation
            features = features.view(N, 1, A).expand(N, C, A)
            onehot = torch.zeros(N, C).cuda()
            onehot.scatter_(1, labels.view(-1, 1), 1)
            NxCxA_onehot = onehot.view(N, C, 1).expand(N, C, A)
            features_by_sort = features.mul(NxCxA_onehot)
            Amount_CXA = NxCxA_onehot.sum(0)
            Amount_CXA[Amount_CXA == 0] = 1
            mean = features_by_sort.sum(0) / Amount_CXA
            sum_weight = onehot.sum(0).view(C, 1).expand(C, A)
            weight = sum_weight.div(
                sum_weight + self.Amount.view(C, 1).expand(C, A)
            )
            weight[sum_weight == 0] = 0
            self.Proto = (self.Proto.mul(1 - weight) + mean.mul(weight)).detach()
            self.Amount = self.Amount + onehot.sum(0)
        else:
            # momentum implementation
            ids_unique = labels.unique()
            for i in ids_unique:
                i = i.item()
                mask_i = (labels == i)
                feature = features[mask_i]
                feature = torch.mean(feature, dim=0)
                self.Amount[i] += len(mask_i)
                self.Proto[i, :] = (1-self.momentum) * feature + self.Proto[i, :] * self.momentum

        
    def save(self, name):
        torch.save({'Proto': self.Proto.cpu(),
                    'Amount': self.Amount.cpu()
                    },
                   os.path.join(self.cfg.OUTPUT_DIR, name))