import numpy as np
from matplotlib import pyplot as plt
from numpy.core.fromnumeric import reshape
# import open3d as o3d
from torch import tensor, Tensor
from typing import List

COLOR_MAP = {
    0: (0., 0., 0.),
    1: (174., 199., 232.),
    2: (152., 223., 138.),
    3: (31., 119., 180.),
    4: (255., 187., 120.),
    5: (188., 189., 34.),
    6: (140., 86., 75.),
    7: (255., 152., 150.),
    8: (214., 39., 40.),
    9: (197., 176., 213.),
    10: (148., 103., 189.),
    11: (196., 156., 148.),
    12: (23., 190., 207.),
    13: (100., 85., 144.),
    14: (247., 182., 210.),
    15: (66., 188., 102.),
    16: (219., 219., 141.),
    17: (140., 57., 197.),
    18: (202., 185., 52.),
    19: (51., 176., 203.),
    20: (200., 54., 131.),
    21: (92., 193., 61.),
    22: (78., 71., 183.),
    23: (172., 114., 82.),
    24: (255., 127., 14.),
    25: (91., 163., 138.),
    26: (153., 98., 156.),
    27: (140., 153., 101.),
    28: (158., 218., 229.),
    29: (100., 125., 154.),
    30: (178., 127., 135.),
    32: (146., 111., 194.),
    33: (44., 160., 44.),
    34: (112., 128., 144.),
    35: (96., 207., 209.),
    36: (227., 119., 194.),
    37: (213., 92., 176.),
    38: (94., 106., 211.),
    39: (82., 84., 163.),
    40: (100., 85., 144.),
}


def visualize_with_label(cloud, labels, window_name="open3d"):
    assert cloud.shape[0] == labels.shape[0]

    pt = o3d.geometry.PointCloud()
    pt.points = o3d.utility.Vector3dVector(cloud.reshape(-1, 3))
    colors = [COLOR_MAP[i] for i in labels]
    colors = np.array(colors) / 255
    pt.colors = o3d.utility.Vector3dVector(colors)

    o3d.visualization.draw_geometries([pt], window_name, width=500, height=500)


def visualize_without_label(cloud, window_name="open3d"):
    if isinstance(cloud, Tensor):
        cloud = cloud.cpu().numpy().reshape((-1, 3))
    pt = o3d.geometry.PointCloud()
    pt.points = o3d.utility.Vector3dVector(cloud)

    o3d.visualization.draw_geometries([pt], window_name, width=500, height=500)


def save_view_point(pcd, filename):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()  # user changes the view and press "q" to terminate
    param = vis.get_view_control().convert_to_pinhole_camera_parameters()
    o3d.io.write_pinhole_camera_parameters(filename, param)
    vis.destroy_window()


def load_view_point(pcd, filename):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    ctr = vis.get_view_control()
    param = o3d.io.read_pinhole_camera_parameters(filename)
    vis.add_geometry(pcd)
    ctr.convert_from_pinhole_camera_parameters(param)
    vis.run()
    vis.destroy_window()


def vis_by_one_viewpoint(load_view_point_flag: bool, window_name: str, pcd):
    # if load_view_point is False:
    # save_view_point = True

    if load_view_point_flag is False:
        save_view_point(pcd=pcd, filename='./view.json')
        print("===> save viewpoint to {}".format("./view.json"))
    elif load_view_point_flag:
        print("===> load viewpoint from {}".format("./view.json"))
        load_view_point(pcd=pcd, filename='./view.json')
    else:
        raise NotImplementedError


def vis(data, label):
    '''
    :param data: n*3的矩阵
    :param label: n*1的矩阵
    :return: 可视化
    '''
    data = data[:, :3]
    labels = np.asarray(label)
    max_label = labels.max()

    # 颜色
    colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))

    pt1 = o3d.geometry.PointCloud()
    pt1.points = o3d.utility.Vector3dVector(data.reshape(-1, 3))
    pt1.colors = o3d.utility.Vector3dVector(colors[:, :3])

    o3d.visualization.draw_geometries([pt1], 'part of cloud', width=500, height=500)






