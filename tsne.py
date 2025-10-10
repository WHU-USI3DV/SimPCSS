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

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

class Tester:
    def __init__(self, cfg, args):
        self.logger = logging.getLogger("PC_DA.tsne")
        self.logger.info("Start testing")

        self.saveres = args.saveres  

        # 哪些层参与特征增强，其特征都经过hb_learner
        self.CL_LAYERS = cfg.distill.CL_BLOCKS

        Model = getattr(models, cfg.model.name)
        self.model = Model(cfg.model.in_feat_size, cfg.model.out_classes, 3, True, CL_LAYERS=self.CL_LAYERS)
        self.classifier = models.classifier(96, cfg.model.out_classes, 3)

        # Single GPU at present
        self.device = torch.device(cfg.pipeline.gpus)
        self.model.to(self.device)
        self.classifier.to(self.device)

        if cfg.pipeline.resume_lb:
            self.logger.info("Loading checkpoint from {}".format(cfg.pipeline.resume_lb))
            self.model = load_model(cfg.pipeline.resume_lb, self.model, False)
            self.classifier = load_model(cfg.pipeline.resume_lb, self.classifier, False)
            
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

        num_per_cls = 50
        feature_D = 96

        # pbar for presenting information in real time
        val_pbar = tqdm(self.test_loader, total=len(self.test_loader))

        vis_cls = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15] 

        hidden_feature = {category: np.zeros([num_per_cls, feature_D]) for category in vis_cls}

        cls_list = []


        with torch.no_grad():
            for cls in vis_cls:
                print('cls:',cls)
                num_now = 0
                is_full = 0
                for i, data in enumerate(val_pbar):
                    coords = data['coordinates'].int().to(self.device)
                    feats = data['features'].float().to(self.device)
                    labels = data['labels'].long().to(self.device)
                    stensor = ME.SparseTensor(features=feats, coordinates=coords)

                    feat, _ = self.model(stensor)
                    # feat = self.classifier(feat)

                    permutation = list(np.random.permutation(len(labels)))
                    features = feat.F.cpu().numpy()[permutation]
                    labels = labels[permutation]

                    for k in range(len(labels)):
                        k_cls = labels[k].item()
                        if k_cls == cls:
                            hidden_feature[cls][num_now,:] = features[k,:]
                            cls_list.append(cls)
                            num_now += 1
                            print(num_now)
                            if num_now == num_per_cls:
                                is_full = 1
                                break

                    if is_full == 1:
                        break
                    else:
                        continue

        print("all ok")

        trainid2name = {
            0: "car",
            1: "bicycle",
            2: "motorcycle",
            3: "truck",
            4: "other-vehicle",
            5: "person",
            6: "rider",
            7: "road",
            8: "sidewalk",
            9: "other-ground",
            10: "building",
            11: "fence",
            12: "vegetation",
            13: "terrain",
            14: "pole",
            15: "traffic-sign"
        }


        embedded = np.concatenate(list(hidden_feature.values()), axis=0)

        tsne = TSNE(n_components=2, init='pca')

        X_embedded = tsne.fit_transform(embedded)
        
        plt.figure(figsize=(16, 16))

        for cl in vis_cls:
            indices = np.where(np.array(cls_list) == cl)
            indices = indices[0]
            plt.scatter(X_embedded[indices, 0], X_embedded[indices, 1], s=10, label=trainid2name[cl])
        plt.legend(bbox_to_anchor=(0.8, 0.8), loc=3, borderaxespad=0)
        plt.show()

                            

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
