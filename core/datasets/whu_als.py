from importlib.resources import is_resource
import os
from re import S
import torch
import yaml
import numpy as np
from tqdm import tqdm
from torch.utils import data
import MinkowskiEngine as ME
import core.datasets.transform as transform
from core.utils import visualization as vis
from configs.config import get_config

ABSOLUTE_PATH = os.path.dirname(os.path.abspath(__file__))


class WhuALS_Dataset(data.Dataset):
    def __init__(
        self,
        mode: str = 'train',
        dataset_path: str = '',
        mapping_path: str = '/home/cz/PC_DA/core/datasets/mapping_cfg/whu-als.yaml',
        voxel_size: float = 0.05,
        use_intensity: bool = False,
        augment_data: list = None,
        sub_num: int = 50000,
        device: str = None,
        num_classes: int = 7,
        ignore_label: int = None
    ):
        self.name = 'WHUALS_Dataset'
        self.maps = yaml.safe_load(open(os.path.join(ABSOLUTE_PATH, mapping_path), 'r'))
        self.ignore_label = ignore_label
        self.use_intensity = use_intensity
        self.mode = mode
        self.augment_data = augment_data
        self.sub_num = sub_num
        self.voxel_size = voxel_size
        self.trans = augment_data
        
        self.split = {'train': ['00', '01', '02', '03', '04', '05', '06', '07', '09', '10'],
                    'val': ['08'],
                    'test': ['08']}

        remap_dict_val = self.maps["learning_map"]
        max_key = max(remap_dict_val.keys())
        remap_lut_val = np.zeros((max_key+1), dtype=np.int32)
        remap_lut_val[list(remap_dict_val.keys())] = list(remap_dict_val.values())
        self.remap_lut_val = remap_lut_val

        self.pcd_path = []
        self.label_path = []
        
        for sequence in self.split[self.mode]:
            num_frames = len(os.listdir(os.path.join(dataset_path, sequence, 'labels')))
            for f in np.arange(num_frames):
                pcd_path = os.path.join(dataset_path, sequence, 'velodyne', f'{int(f):06d}.bin')
                label_path = os.path.join(dataset_path, sequence, 'labels', f'{int(f):06d}.label')
                self.pcd_path.append(pcd_path)
                self.label_path.append(label_path)

        self.color_map = np.array([(255, 255, 255),  # unlabelled
                                    (25, 25, 255),  # car
                                    (187, 0, 255),  # bicycle
                                    (187, 50, 255),  # motorcycle
                                    (0, 247, 255),  # truck
                                    (50, 162, 168),  # other-vehicle
                                    (250, 178, 50),  # person
                                    (255, 196, 0),  # bicyclist
                                    (255, 196, 0),  # motorcyclist
                                    (0, 0, 0),  # road
                                    (148, 148, 148),  # parking
                                    (255, 20, 60),  # sidewalk
                                    (164, 173, 104),  # other-ground
                                    (233, 166, 250),  # building
                                    (255, 214, 251),  # fence
                                    (157, 234, 50),  # vegetation
                                    (107, 98, 56),  # trunk
                                    (78, 72, 44),  # terrain
                                    (83, 93, 130),  # pole
                                    (173, 23, 121)])/255.   # traffic-sign

    def __len__(self):
        return len(self.pcd_path)
            
    def load_label_kitty(self, label_path: str):
        label = np.fromfile(label_path, dtype=np.uint32)
        label = label.reshape((-1))
        sem_label = label & 0xFFFF  # semantic label in lower half
        inst_label = label >> 16  # instance id in upper half
        assert ((sem_label + (inst_label << 16) == label).all())
        sem_label = self.remap_lut_val[sem_label]
        return sem_label.astype(np.int32)

    def get_dataset_weights(self):
        num_class = np.zeros(self.remap_lut_val.max()+1)
        for i in tqdm(range(len(self.label_path)), desc='loading weights', leave=True):
            label_tmp = self.label_path[i]
            label = self.load_label_kitty(label_tmp)
            lbl, count = np.unique(label, return_counts=True)
            if self.ignore_label is not None:
                if self.ignore_label in lbl:
                    count = count[lbl != self.ignore_label]
                    lbl = lbl[lbl != self.ignore_label]
            num_class[lbl] += count
        weights = num_class/sum(num_class)
        inv_weights = 1 / (weights + 0.02)
        return inv_weights

    def get_data(self, i: int):
        pcd_tmp = self.pcd_path[i]
        label_tmp = self.label_path[i]

        pcd = np.fromfile(pcd_tmp, dtype=np.float32).reshape((-1, 4))
        label = self.load_label_kitty(label_tmp)
        points = pcd[:, :3]
        if self.use_intensity:
            colors = points[:, 3][..., None]
        else:
            colors = np.ones((points.shape[0], 1), dtype=np.float32)
        data = {'points': points, 'colors': 'colors', 'labels': label}

        points = data['points']
        colors = data['colors']
        labels = data['labels']

        return {'coordinates': points,
        'features': colors,
        'labels': labels,
        'idx': i
        }

    def __getitem__(self, i: int):
        pcd_tmp = self.pcd_path[i]
        label_tmp = self.label_path[i]

        pcd = np.fromfile(pcd_tmp, dtype=np.float32).reshape((-1, 4))
        label = self.load_label_kitty(label_tmp)
        points = pcd[:, :3]
        if self.use_intensity:
            colors = pcd[:, 3][..., None]
        else:
            colors = np.ones((points.shape[0], 1), dtype=np.float32)
        data = {'points': points, 'colors': colors, 'labels': label}

        points = data['points']
        colors = data['colors']
        labels = data['labels']

        # if self.mode == "train"  and self.trans:
        if self.trans:
            if 'random_sample' in self.trans:
                sample_idx = transform.random_sample(points, self.sub_num)
                points = points[sample_idx]
                colors = colors[sample_idx]
                labels = labels[sample_idx]
            if 'z_rotate' in self.trans:
                points = transform.random_rotate(points, 'z')

        if self.ignore_label is None:
            vox_ign_label = -100
        else:
            vox_ign_label = self.ignore_label

        quantized_coords, feats, labels, voxel_idx = ME.utils.sparse_quantize(
            points,
            colors,
            labels=labels,
            ignore_label=vox_ign_label,
            quantization_size=self.voxel_size,
            return_index=True
        )

        if isinstance(quantized_coords, np.ndarray):
            quantized_coords = torch.from_numpy(quantized_coords)

        if isinstance(feats, np.ndarray):
            feats = torch.from_numpy(feats)

        if isinstance(labels, np.ndarray):
            labels = torch.from_numpy(labels)

        if isinstance(voxel_idx, np.ndarray):
            voxel_idx = torch.from_numpy(voxel_idx)

        return {
            "coordinates": quantized_coords,
            "features": feats,
            "labels": labels,
            "idx": torch.tensor(i)
        }


if __name__ == "__main__":
    config_file = '/home/cz/PC_DA/configs/source/WhuALS.yaml'
    cfg = get_config(config_file)
    dataset = WhuALS_Dataset(
        mode='train',
        dataset_path=cfg.dataset.dataset_path,
        mapping_path=cfg.dataset.mapping_path,
        voxel_size=cfg.dataset.voxel_size,
        use_intensity=cfg.dataset.use_intensity,
        augment_data=[],
        sub_num=cfg.dataset.sub_numpts,
        num_classes=cfg.model.out_classes,
        ignore_label=cfg.dataset.ignore_label,
    )
    pts_path = "/home/cz/data/whu_als/00/velodyne/000000.bin"
    labels_path = "/home/cz/data/whu_als/00/labels/000000.label"
    pts = np.fromfile(pts_path, dtype=np.float32).reshape((-1, 4))[:, 0:3]
    labels_vis = dataset.load_label_kitty(labels_path)

    sample_idx = transform.random_sample(pts, 80000)
    pts = pts[sample_idx]
    labels_vis = labels_vis[sample_idx]

    vis.visualize_with_label(pts, labels_vis, window_name="open3d")




