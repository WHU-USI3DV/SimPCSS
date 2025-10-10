from importlib.resources import is_resource
import os
from re import S
import torch
import yaml
import numpy as np
from tqdm import tqdm
from torch.utils import data
import MinkowskiEngine as ME
from . import transform
import pickle
from core.utils.metric import mkdir

ABSOLUTE_PATH = os.path.dirname(os.path.abspath(__file__))


class SemanticKittiDataset(data.Dataset):
    def __init__(
        self,
        mode: str = 'train',
        dataset_path: str = '',
        mapping_path: str = '',
        voxel_size: float = 0.05,
        use_intensity: bool = False,
        augment_data: list = None,
        sub_num: int = 50000,
        device: str = None,
        num_classes: int = 7,
        ignore_label: int = None,
        dataset_name: str = '',
        return_blocks: dict = {}
    ):
        self.name = 'SemanticKittiDataset'
        self.maps = yaml.safe_load(open(os.path.join(ABSOLUTE_PATH, mapping_path), 'r'))
        self.ignore_label = ignore_label
        self.use_intensity = use_intensity
        self.mode = mode
        self.augment_data = augment_data
        self.sub_num = sub_num
        self.voxel_size = voxel_size
        self.trans = augment_data
        self.dataset_name = dataset_name  
        self.return_blocks = return_blocks
        
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
    

        self.class2name = np.array(['car', 'bicycle', 'motorcycle', 'truck', 'other-vehicle',
                             'person', 'bicyclist', 'motorcyclist',
                             'road', 'parking', 'sidewalk', 'other-ground',
                             'building', 'fence', 'vegetation', 'trunk',
                             'terrain', 'pole', 'traffic-sign'])

        # # RGB 19
        # self.class2color = np.array([(0, 0, 0),  # unlabelled
        #                             (100, 150, 245),  # car
        #                             (100, 230, 245),  # bicycle
        #                             (30, 60, 150),  # motorcycle
        #                             (80, 30, 180),  # truck
        #                             (0, 0, 255),  # other-vehicle
        #                             (255, 30, 30),  # person
        #                             (255, 40, 200),  # bicyclist
        #                             (150, 30, 90),  # motorcyclist
        #                             (255, 0, 255),  # road
        #                             (255, 150, 255),  # parking
        #                             (75, 0, 75),  # sidewalk
        #                             (175, 0, 75),  # other-ground
        #                             (255, 200, 0),  # building
        #                             (255, 120, 50),  # fence
        #                             (0, 175, 0),  # vegetation
        #                             (135, 60, 0),  # trunk
        #                             (150, 240, 80),  # terrain
        #                             (255, 140, 150),  # pole
        #                             (255, 0, 0)])/255.   # traffic-sign
        # RGB 16
        self.class2color = np.array([(255, 255, 255),  # unlabelled
                        (100, 150, 245),  # car
                        (100, 230, 245),  # bicycle
                        (30, 60, 150),  # motorcycle
                        (80, 30, 180),  # truck
                        (0, 0, 255),  # other-vehicle
                        (255, 30, 30),  # person
                        (255, 40, 200),  # rider
                        (255, 0, 255),  # road
                        (75, 0, 75),  # sidewalk
                        (175, 0, 75),  # other-ground
                        (255, 200, 0),  # building
                        (255, 120, 50),  # fence
                        (0, 175, 0),  # vegetation
                        (150, 240, 80),  # terrain
                        (255, 140, 150),  # pole
                        (255, 0, 0)]) / 255.  # traffic-sign

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
        weights_dir = os.path.join(ABSOLUTE_PATH, '_weights')
        mkdir(weights_dir) if not os.path.exists(weights_dir) else None
        weights_path = os.path.join(weights_dir, self.dataset_name + '.pkl')
        if not os.path.exists(weights_path):
            num_class = np.zeros(self.remap_lut_val.max() + 1)
            for i in tqdm(range(len(self.label_path)), desc='loading weights', leave=True):
                label_tmp = self.label_path[i]
                label = self.load_label_kitty(label_tmp)
                lbl, count = np.unique(label, return_counts=True)
                if self.ignore_label is not None:
                    if self.ignore_label in lbl:
                        count = count[lbl != self.ignore_label]
                        lbl = lbl[lbl != self.ignore_label]
                num_class[lbl] += count
            weights = num_class / sum(num_class)
            # inv_weights = 1 / (weights + 0.02)
            inv_weights = 1 / weights ** 0.5
            print('>>>>>>>>>>>>>>>> Load dataset weights >>>>>>>>>>>>>>>>')
            print(inv_weights)
            with open(weights_path, 'wb') as f:
                pickle.dump(inv_weights, f)
        else:
            print('>>>>>>>>>>>>>>>> Load dataset weights >>>>>>>>>>>>>>>>')
            with open(weights_path, 'rb') as f:
                inv_weights = pickle.load(f)
                print(inv_weights)
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

        if self.mode == "train" and self.trans:
            if 'random_sample' in self.trans:
                sample_idx = transform.random_sample(points, self.sub_num)
                points = points[sample_idx]
                colors = colors[sample_idx]
                labels = labels[sample_idx]
            if 'z_rotate' in self.trans:
                points = transform.random_rotate(points, 'z')
            if 'random_scale' in self.trans:
                points = transform.random_scale(points)
            if 'random_flip' in self.trans:
                points = transform.random_flip(points)
            if 'random_jitter' in self.trans:
                points = transform.random_jitter(points)
        
        if self.ignore_label is None:
            vox_ign_label = -100
        else:
            vox_ign_label = self.ignore_label

        quantized_coords, feats_vox, labels_vox, voxel_idx = ME.utils.sparse_quantize(
            points,
            colors,
            labels=labels,
            ignore_label=vox_ign_label,
            quantization_size=self.voxel_size,
            return_index=True
        )

        # labels_down = {}
        # stride_mapping = {'block4': 16, 'block5': 8, 'block6': 4, 'block7': 2}
        # for k, v in self.return_blocks.items():
        #     if k=='block8':
        #         continue
        #     downsample_rate = stride_mapping[k]
        #     quantized_coords_downsample, labels_vox_downsample, voxel_idx_downsample = ME.utils.sparse_quantize(
        #         points,
        #         labels=labels,
        #         ignore_label=vox_ign_label,
        #         quantization_size=self.voxel_size * downsample_rate,
        #         return_index=True
        #     )
        #     labels_down.update({k: labels_vox_downsample})

        if isinstance(quantized_coords, np.ndarray):
            quantized_coords = torch.from_numpy(quantized_coords)

        if isinstance(feats_vox, np.ndarray):
            feats_vox = torch.from_numpy(feats_vox)

        if isinstance(labels_vox, np.ndarray):
            labels_vox = torch.from_numpy(labels_vox)

        if isinstance(voxel_idx, np.ndarray):
            voxel_idx = torch.from_numpy(voxel_idx)

        # for k, v in labels_down.items():
        #     if isinstance(v, np.ndarray):
        #         labels_down[k] = torch.from_numpy(v)

        return {
            "coordinates": quantized_coords,
            "features": feats_vox,
            "labels": labels_vox,
            "idx": torch.tensor(i),
            # "labels_down": labels_down,
        }
