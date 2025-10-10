import os
import numpy as np
from core.datasets.SemanticKitti import SemanticKittiDataset
from core.datasets.whu_als import WhuALS_Dataset
from core.datasets.synlidar import SynLiDARDataset
from core.datasets.mspclidar import MspcLiDARDataset
from core.datasets.nuScenens import nuScenesDataset


def get_dataset(
        cfg,
        mode,
        is_source: bool = True,
        return_blocks: dict = {}
):
    '''
        :param dataset_name: name of the dataset
        :param dataset_path: absolute path to data
        :param mode: dataset for training, validation or testing
        :param voxel_size: voxel size for voxelization
        :param use_intensity: use intensity as feature or not
        :param augment_data: methods for augmentation
        :param num_classes: number of classes considered
        :param ignore_label: label to ignore
        :param mapping_path: path to mapping files for labels
        :param is_source: is the dataset from source or target domain
    '''

    trans = []

    if mode == 'train':
        if is_source:
            if cfg.source_dataset.sub_numpts is not None:
                trans = ["random_sample"] + trans
            if cfg.source_dataset.augmentation.z_rotate == True:
                trans = ["z_rotate"] + trans
            if cfg.source_dataset.augmentation.random_scale == True:
                trans = ["random_scale"] + trans
            if cfg.source_dataset.augmentation.random_flip == True:
                trans = ["random_flip"] + trans
            if cfg.source_dataset.augmentation.random_jitter == True:
                trans = ["random_jitter"] + trans
        else:
            if cfg.target_dataset.sub_numpts is not None:
                trans = ["random_sample"] + trans
            if cfg.target_dataset.augmentation.z_rotate == True:
                trans = ["z_rotate"] + trans
            if cfg.target_dataset.augmentation.random_scale == True:
                trans = ["random_scale"] + trans
            if cfg.target_dataset.augmentation.random_flip == True:
                trans = ["random_flip"] + trans
            if cfg.target_dataset.augmentation.random_jitter == True:
                trans = ["random_jitter"] + trans

    if is_source:
        if cfg.source_dataset.name == 'SemanticKitti':
            dataset = SemanticKittiDataset(
                mode=mode, dataset_path=cfg.source_dataset.dataset_path, mapping_path=cfg.source_dataset.mapping_path,
                voxel_size=cfg.source_dataset.voxel_size, use_intensity=cfg.source_dataset.use_intensity,
                augment_data=trans, sub_num=cfg.source_dataset.sub_numpts, num_classes=cfg.model.out_classes,
                ignore_label=cfg.model.ignore_label,
                dataset_name = cfg.source_dataset.name,
                return_blocks = return_blocks,
            )

        elif cfg.source_dataset.name == 'nuScenes':
            dataset = nuScenesDataset(
                mode=mode, dataset_path=cfg.source_dataset.dataset_path, mapping_path=cfg.source_dataset.mapping_path,
                voxel_size=cfg.source_dataset.voxel_size, use_intensity=cfg.source_dataset.use_intensity,
                augment_data=trans, sub_num=cfg.source_dataset.sub_numpts, num_classes=cfg.model.out_classes,
                ignore_label=cfg.model.ignore_label,
                dataset_name = cfg.source_dataset.name,
            )

        elif cfg.source_dataset.name == 'SynLiDAR':
            dataset = SynLiDARDataset(
                mode=mode, dataset_path=cfg.source_dataset.dataset_path, mapping_path=cfg.source_dataset.mapping_path,
                voxel_size=cfg.source_dataset.voxel_size, use_intensity=cfg.source_dataset.use_intensity,
                augment_data=trans, sub_num=cfg.source_dataset.sub_numpts, num_classes=cfg.model.out_classes,
                ignore_label=cfg.model.ignore_label,
                dataset_name = cfg.source_dataset.name,
            )

        elif 'MspcLiDAR' in cfg.source_dataset.name:
                dataset = MspcLiDARDataset(
                mode=mode, dataset_path=cfg.source_dataset.dataset_path, mapping_path=cfg.source_dataset.mapping_path,
                voxel_size=cfg.source_dataset.voxel_size, use_intensity=cfg.source_dataset.use_intensity,
                augment_data=trans,sub_num=cfg.source_dataset.sub_numpts, num_classes=cfg.model.out_classes,
                ignore_label=cfg.model.ignore_label,
                dataset_name = cfg.source_dataset.name,
                return_blocks = return_blocks,
            )
    else:
        if cfg.target_dataset.name == 'SemanticKitti':
            dataset = SemanticKittiDataset(
                mode=mode, dataset_path=cfg.target_dataset.dataset_path, mapping_path=cfg.target_dataset.mapping_path,
                voxel_size=cfg.target_dataset.voxel_size, use_intensity=cfg.target_dataset.use_intensity,
                augment_data=trans, sub_num=cfg.target_dataset.sub_numpts, num_classes=cfg.model.out_classes,
                ignore_label=cfg.model.ignore_label,
                dataset_name = cfg.target_dataset.name,
                return_blocks = return_blocks,
            )

        elif cfg.target_dataset.name == 'nuScenes':
            dataset = nuScenesDataset(
                mode=mode, dataset_path=cfg.target_dataset.dataset_path, mapping_path=cfg.target_dataset.mapping_path,
                voxel_size=cfg.target_dataset.voxel_size, use_intensity=cfg.target_dataset.use_intensity,
                augment_data=trans, sub_num=cfg.target_dataset.sub_numpts, num_classes=cfg.model.out_classes,
                ignore_label=cfg.model.ignore_label,
                dataset_name = cfg.target_dataset.name,
            )

        elif cfg.target_dataset.name == 'SynLiDAR':
            dataset = SynLiDARDataset(
                mode=mode, dataset_path=cfg.target_dataset.dataset_path, mapping_path=cfg.target_dataset.mapping_path,
                voxel_size=cfg.target_dataset.voxel_size, use_intensity=cfg.target_dataset.use_intensity,
                augment_data=trans, sub_num=cfg.target_dataset.sub_numpts, num_classes=cfg.model.out_classes,
                ignore_label=cfg.model.ignore_label,
                dataset_name = cfg.target_dataset.name,
            )

        elif 'MspcLiDAR' in cfg.target_dataset.name:
            dataset = MspcLiDARDataset(
                mode=mode, dataset_path=cfg.target_dataset.dataset_path, mapping_path=cfg.target_dataset.mapping_path,
                voxel_size=cfg.target_dataset.voxel_size, use_intensity=cfg.target_dataset.use_intensity,
                augment_data=trans,sub_num=cfg.target_dataset.sub_numpts, num_classes=cfg.model.out_classes,
                ignore_label=cfg.model.ignore_label,
                dataset_name = cfg.target_dataset.name,
                return_blocks = return_blocks,
            )

    return dataset
