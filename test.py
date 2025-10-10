import os
import argparse
import logging
import numpy as np
from tqdm import tqdm
from sklearn.metrics import jaccard_score

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import MinkowskiEngine as ME

from configs.config import get_config
from core.utils.metric import mkdir, AverageMeter, intersectionAndUnionGPU, load_model
from core.datasets.build import get_dataset
from core.utils.logger import setup_logger
from core.models.collation import CollateFN
import core.models as models
import open3d as o3d


class Tester:
    def __init__(self, cfg, args):
        self.logger = logging.getLogger("PC_DA.tester")
        self.logger.info("Start testing")

        self.saveres = args.saveres

        Model = getattr(models, cfg.model.name)
        self.model = Model(cfg.model.in_feat_size, cfg.model.out_classes)
        self.classifier = models.classifier(96, cfg.model.out_classes, 3)

        # Single GPU at present
        self.device = torch.device(cfg.pipeline.gpus)
        self.model.to(self.device)
        self.classifier.to(self.device)

        if cfg.pipeline.resume_hb:
            self.logger.info("Loading checkpoint from {}".format(cfg.pipeline.resume_hb))
            self.model = load_model(cfg.pipeline.resume_hb, 'model', self.model, False)
            self.classifier = load_model(cfg.pipeline.resume_hb, 'model', self.classifier, False)
            
        self.test_data = get_dataset(cfg, mode='val', is_source=args.test_source)

        collation = CollateFN()

        self.test_loader = DataLoader(
            self.test_data,
            collate_fn = collation,
            batch_size = cfg.src_dataloader.batch_size if args.test_source else cfg.tgt_dataloader.batch_size,
            shuffle = False,
            num_workers = cfg.src_dataloader.num_workers if args.test_source else cfg.tgt_dataloader.num_workers,
            pin_memory = True
        )


    def test(self, cfg, args):
        self.logger = logging.getLogger("PC_DA.tester")

        self.model.eval()

        torch.cuda.empty_cache()

        intersection_meter = AverageMeter()
        union_meter = AverageMeter()
        target_meter = AverageMeter()

        # pbar for presenting information in real time
        val_pbar = tqdm(self.test_loader, total=len(self.test_loader))

        with torch.no_grad():
            for i, data in enumerate(val_pbar):
                coords = data['coordinates'].int().to(self.device)
                feats = data['features'].float().to(self.device)
                labels = data['labels'].long().to(self.device)
                stensor = ME.SparseTensor(features=feats, coordinates=coords)

                feat, _ = self.model(stensor)
                out = self.classifier(feat).F
                pred = out.max(dim=-1).indices

                intersection, union, target = intersectionAndUnionGPU(
                    pred, labels, cfg.model.out_classes, cfg.model.ignore_label
                )

                intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
                intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)

                if self.saveres:
                    coords = data["coordinates"].cpu()
                    labels = data["labels"].cpu()
                    pred = pred.cpu()

                    batch_size = torch.unique(coords[:,0]).max()+1
                    sample_idx = data["idx"]
                    for b in range(batch_size.int()):
                        s_idx = int(sample_idx[b].item())
                        b_idx = coords[:, 0] == b
                        points = coords[b_idx, 1:]
                        p = pred[b_idx]
                        l = labels[b_idx]

                        pcd = o3d.geometry.PointCloud()
                        pcd.points = o3d.utility.Vector3dVector(points)
                        pcd.colors = o3d.utility.Vector3dVector(self.test_data.class2color[p+1])

                        iou_tmp = jaccard_score(p.cpu().numpy(), l.cpu().numpy(), average=None,
                        labels=np.arange(0, cfg.model.out_classes), zero_division='warn')

                        present_labels, _ = np.unique(l.cpu().numpy(), return_counts=True)
                        present_labels = present_labels[present_labels != cfg.model.ignore_label]
                        iou_tmp = np.nanmean(iou_tmp[present_labels]) * 100

                        save_preds_path = os.path.join(args.saveres, self.test_data.dataset_name + '_preds')
                        save_labels_path = os.path.join(args.saveres, self.test_data.dataset_name + '_labels')
                        os.makedirs(save_preds_path, exist_ok=True)
                        os.makedirs(save_labels_path, exist_ok=True)
                        print(args.saveres, 'preds', f'{s_idx}_{int(iou_tmp)}.ply')

                        o3d.io.write_point_cloud(os.path.join(save_preds_path, f'{s_idx}_{int(iou_tmp)}.ply'), pcd)

                        pcd.colors = o3d.utility.Vector3dVector(self.test_data.class2color[l+1])

                        o3d.io.write_point_cloud(os.path.join(save_labels_path, f'{s_idx}.ply'), pcd)

                if i % 5 == 0:
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
    parser = argparse.ArgumentParser(description="Start testing")
    parser.add_argument(        
        "-cfg",
        "--config_file",
        default="",
        type=str,
        help="Path to config file"
        )

    parser.add_argument(
        '--saveres', 
        type=str,
        help="Path save the segmentation result"
        )

    parser.add_argument(
        '--test_source',
        type=bool,
        default=False,
        help="Test PointCloud from source or target"
    )

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

    tester = Tester(cfg, args)
    tester.test(cfg, args)

if __name__ == '__main__':
    main()
