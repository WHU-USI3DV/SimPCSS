import torch
import numpy as np
import random
from time import time


def random_sample(points, sub_num):
    """
    :param points: input points of shape [N, 3]
    :param center: center to sample around, default is None, not used for now
    :return: np.ndarray of N points sampled from input points
    """
    num_points = points.shape[0]

    if sub_num is not None:
        if sub_num <= num_points:
            sampled_idx = np.random.choice(np.arange(num_points), sub_num, replace=False)
        else:
            over_idx = np.random.choice(np.arange(num_points), sub_num - num_points, replace=True)
            sampled_idx = np.concatenate([np.arange(num_points), over_idx])
    else:
        sampled_idx = np.arange(num_points)

    return sampled_idx


def rotate_shape(x, axis, angle):
    """
    Input:
        x: pointcloud data, [B, C, N]
        axis: axis to do rotation about
        angle: rotation angle
    Return:
        A rotated shape
    """
    R_x = np.asarray([[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]])
    R_y = np.asarray([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])
    R_z = np.asarray([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])

    if axis == "x":
        return x.dot(R_x).astype('float32')
    elif axis == "y":
        return x.dot(R_y).astype('float32')
    else:
        return x.dot(R_z).astype('float32')


def random_rotate(PC, axis):
    """
    Apply random rotation about one axis
    Input:
        x: pointcloud data, [B, C, N]
        axis: axis to do rotation about
    Return:
        A rotated shape
    """
    rotation_angle = 2 * np.pi * np.random.uniform(0.0, 1.0)
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    if axis == 'x':
        R_x = [[1, 0, 0], [0, cosval, -sinval], [0, sinval, cosval]]
        X = PC @ R_x
    elif axis == 'y':
        R_y = [[cosval, 0, sinval], [0, 1, 0], [-sinval, 0, cosval]]
        X = PC @ R_y
    elif axis == 'z':
        R_z = [[cosval, -sinval, 0], [sinval, cosval, 0], [0, 0, 1]]
        X = PC @ R_z
    return X


def random_rotate_GPU(PC, axis):
    rotation_angle = 2 * np.pi * np.random.uniform(0.0, 1.0)
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    if axis == 'x':
        R_x = [[1, 0, 0], [0, cosval, -sinval], [0, sinval, cosval]]
        R_x = torch.tensor(R_x).cuda().type(torch.float32)
        PC = torch.from_numpy(PC.astype(np.float32)).cuda()
        X = PC @ R_x
    elif axis == 'y':
        R_y = [[cosval, 0, sinval], [0, 1, 0], [-sinval, 0, cosval]]
        R_y = torch.tensor(R_y).cuda().type(torch.float32)
        PC = torch.from_numpy(PC.astype(np.float32)).cuda()
        X = PC @ R_y
    elif axis == 'z':
        R_z = [[cosval, -sinval, 0], [sinval, cosval, 0], [0, 0, 1]]
        R_z = torch.tensor(R_z).cuda().type(torch.float32)
        PC = torch.from_numpy(PC.astype(np.float32)).cuda()
        X = PC @ R_z
    return X.cpu().numpy()

    
def translate_pointcloud(PC):
    """
    Input:
        PC: pointcloud data, [B, C, N]
    Return:
        A translated shape
    """
    xyz1 = np.random.uniform(low=2. / 3., high=3. / 2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])

    translated_pointcloud = np.add(np.multiply(PC, xyz1), xyz2).astype('float32')
    return translated_pointcloud


def random_scale(PC, scale_range: list = [0.95, 1.05]):
    scale_factor = np.random.uniform(scale_range[0], scale_range[1])
    PC = PC * scale_factor
    return PC


def random_scale_GPU(PC, scale_range: list = [0.95, 1.05]):
    scale_factor = np.random.uniform(scale_range[0], scale_range[1])
    scale_factor = torch.tensor(scale_factor).cuda()
    PC = torch.from_numpy(PC).cuda()
    PC = PC * scale_factor
    return PC.cpu().numpy()

def random_flip(PC):
    if np.random.rand() < 0.5:
        PC[:,0] = -PC[:, 0]
    if np.random.rand() < 0.5:
        PC[:,1] = -PC[:, 1]
    # flip_type = np.random.choice(4, 1)
    # if flip_type == 1:
    #     PC[:,0] = -PC[:, 0]
    # if flip_type == 2:
    #     PC[:,1] = -PC[:, 1]
    # if flip_type == 3:
    #     PC[:,2] = -PC[:, 2]
    return PC


def random_jitter(pc):
    sigma = 0.005
    clip = 0.02
    jitter = np.clip(
        sigma * np.random.randn(pc.shape[0], 3),
        -clip,
        clip,
    )
    pc += jitter
    return pc