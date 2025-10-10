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

ABSOLUTE_PATH = os.path.dirname(os.path.abspath(__file__))


class SynLiDARDataset(data.Dataset):
    def __init__(
        self,
        mode: str = 'train',
        dataset_path: str = '',
        mapping_path: str = '',
        voxel_size: float = 0.05,
        use_intensity: bool = False,
        augment_data: list = None,
        sub_num: int = 50000,
        num_classes: int = 7,
        ignore_label: int = None,
        version: str = 'full',
        device: str = None,
        dataset_name: str = ''
    ):
        self.name = 'SynLiDARDataset'
        self.maps = yaml.safe_load(open(os.path.join(ABSOLUTE_PATH, mapping_path), 'r'))
        self.ignore_label = ignore_label
        self.use_intensity = use_intensity
        self.mode = mode
        self.augment_data = augment_data
        self.sub_num = sub_num
        self.voxel_size = voxel_size
        self.trans = augment_data
        self.dataset_path = dataset_path
        self.version = version

        if self.version == 'full':
            self.sequences = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
            self.get_splits()
        elif self.version == 'mini':
            self.sequences = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
            self.get_splits()
        elif self.version == 'sequential':
            self.sequences = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
            self.get_splits()
        else:
            raise NotImplementedError

        remap_dict_val = self.maps["learning_map"]
        max_key = max(remap_dict_val.keys())
        remap_lut_val = np.zeros((max_key+1), dtype=np.int32)
        remap_lut_val[list(remap_dict_val.keys())] = list(remap_dict_val.values())
        self.remap_lut_val = remap_lut_val

        self.pcd_path = []
        self.label_path = []
        self.selected = []
        
        for sequence, frames in self.split[self.mode].items():
            for frame in frames:
                pcd_path = os.path.join(self.dataset_path, sequence, 'velodyne', f'{int(frame):06d}.bin')
                label_path = os.path.join(self.dataset_path, sequence, 'labels', f'{int(frame):06d}.label')
                self.pcd_path.append(pcd_path)
                self.label_path.append(label_path)

        print(f'--> Selected {len(self.pcd_path)} for {self.mode}')

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

    def get_splits(self):
        if self.version == 'full':
            split_path = os.path.join(ABSOLUTE_PATH, '_splits/synlidar.pkl')
        elif self.version == 'mini':
            split_path = os.path.join(ABSOLUTE_PATH, '_splits/synlidar_mini.pkl')
        elif self.version == 'sequential':
            split_path = os.path.join(ABSOLUTE_PATH, '_splits/synlidar_sequential.pkl')
        else:
            raise NotImplementedError

        if not os.path.isfile(split_path):
            self.split = {'train': {s: [] for s in self.sequences},
                          'val': {s: [] for s in self.sequences}}
            if self.version != 'sequential':
                for sequence in self.sequences:
                    sequence_path = os.path.join(self.dataset_path, sequence, 'velodyne')
                    num_frames = len(os.listdir(sequence_path))
                    # num_frames = len(os.listdir(os.path.join(self.dataset_path, sequence, 'labels')))
                    valid_frames = []

                    for frame in os.listdir(sequence_path):
                        filename =  os.path.splitext(frame)[0]
                        pcd_path = os.path.join(self.dataset_path, sequence, 'velodyne', filename) + '.bin'
                        label_path = os.path.join(self.dataset_path, sequence, 'labels', filename) + '.label'

                        if os.path.isfile(pcd_path) and os.path.isfile(label_path):
                            valid_frames.append(filename)
                    if self.version == 'full':
                        train_selected = np.random.choice(valid_frames, int(num_frames * 0.8), replace=False)
                    else:
                        train_selected = np.random.choice(valid_frames, int(num_frames * 0.2), replace=False)

                    for t in train_selected:
                        valid_frames.remove(t)

                    validation_selected = np.random.choice(valid_frames, int(num_frames * 0.2), replace=False)

                    self.split['train'][sequence].extend(train_selected)
                    self.split['val'][sequence].extend(validation_selected)
                torch.save(self.split, split_path)
            else:
                for sequence in self.sequences:
                    num_frames = len(os.listdir(os.path.join(self.dataset_path, sequence, 'labels')))
                    total_for_sequence = int(num_frames/10)
                    print('--> TOTAL:', total_for_sequence)
                    train_selected = []
                    validation_selected = []

                    for v in np.arange(num_frames):
                        pcd_path = os.path.join(self.dataset_path, sequence, 'velodyne', f'{int(v):06d}.bin')
                        label_path = os.path.join(self.dataset_path, sequence, 'labels', f'{int(v):06d}.label')

                        if os.path.isfile(pcd_path) and os.path.isfile(label_path):
                            if len(train_selected) == 0:
                                train_selected.append(v)
                                last_added = v
                            elif len(train_selected) < total_for_sequence and (v-last_added) >= 5:
                                train_selected.append(v)
                                last_added = v
                                print(last_added)
                            else:
                                validation_selected.append(v)

                    validation_selected = np.random.choice(validation_selected, int(num_frames/100), replace=False)

                    self.split['train'][sequence].extend(train_selected)
                    self.split['validation'][sequence].extend(validation_selected)
                torch.save(self.split, split_path)

        else:
            self.split = torch.load(split_path)
            print('SEQUENCES', self.split.keys())
            print('TRAIN SEQUENCES', self.split['train'].keys())

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
            inv_weights = 1 / (weights + 0.02)
            with open(weights_path, 'wb') as f:
                pickle.dump(inv_weights, f)
        else:
            print('>>>>>>>>>>>>>>>> Load dataset weights >>>>>>>>>>>>>>>>')
            with open(weights_path, 'rb') as f:
                inv_weights = pickle.load(f)
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



