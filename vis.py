import numpy as np
import open3d as o3d
import matplotlib as plt
import os

COLOR_MAP = {
    0: (0, 0, 0),  # unlabelled
    1: (255, 0, 255),  # road
    2: (75, 0, 75),  # sidewalk
    3: (255, 200, 0),  # building
    4: (255, 120, 50),  # fence
    5: (255, 120, 50),  # fence
    6: (255, 140, 150),  # pole
    7: (255, 0, 0),  # traffic-sign
    8: (255, 0, 0),  # traffic-sign
    9: (0, 175, 0),  # vegetation
    10: (150, 240, 80),  # terrain
    11: (0, 0, 0),  # unlabelled
    12: (255, 30, 30),  # person
    13: (255, 40, 200),  # rider
    14: (100, 150, 245),  # car
    15: (80, 30, 180),  # truck
    16: (0, 0, 255),  # other-vehicle
    17: (0, 0, 255),  # other-vehicle
    18: (30, 60, 150),  # motorcycle
    19: (100, 230, 245),  # bicycle
    20: (0, 0, 0),  # unlabelled
    21: (0, 0, 0),  # unlabelled
    22: (0, 0, 0),  # unlabelled
    23: (0, 0, 0),  # unlabelled
    24: (255, 0, 255),  # road
    25: (175, 0, 75),  # other-ground
    26: (0, 0, 0),  # unlabelled
    27: (0, 0, 0),  # unlabelled
    28: (0, 0, 0),  # unlabelled
}

COLOR_MAP_KITTI = {
    0: (0., 0., 0.),
    1: (174., 199., 232.),
    10: (152., 223., 138.),
    11: (31., 119., 180.),
    13: (255., 187., 120.),
    15: (188., 189., 34.),
    16: (140., 86., 75.),
    18: (255., 152., 150.),
    20: (214., 39., 40.),
    30: (197., 176., 213.),
    31: (148., 103., 189.),
    32: (196., 156., 148.),
    40: (23., 190., 207.),
    44: (100., 85., 144.),
    48: (247., 182., 210.),
    49: (66., 188., 102.),
    50: (219., 219., 141.),
    51: (140., 57., 197.),
    52: (202., 185., 52.),
    60: (51., 176., 203.),
    70: (200., 54., 131.),
    71: (92., 193., 61.),
    72: (78., 71., 183.),
    80: (172., 114., 82.),
    81: (255., 127., 14.),
    99: (91., 163., 138.),
    252: (153., 98., 156.),
    253: (140., 153., 101.),
    254: (158., 218., 229.),
    255: (100., 125., 154.),
    256: (178., 127., 135.),
    257: (135., 127., 178.),
    258: (146., 111., 194.),
    259: (44., 160., 44.),
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
    colors = plt.cm.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))

    pt1 = o3d.geometry.PointCloud()
    pt1.points = o3d.utility.Vector3dVector(data.reshape(-1, 3))
    pt1.colors = o3d.utility.Vector3dVector(colors[:, :3])

    o3d.visualization.draw_geometries([pt1], 'part of cloud', width=500, height=500)


def load_label_kitti(label_path: str):
    label = np.fromfile(label_path, dtype=np.uint32)
    label = label.reshape((-1))
    sem_label = label & 0xFFFF  # semantic label in lower half
    inst_label = label >> 16  # instance id in upper half
    assert ((sem_label + (inst_label << 16) == label).all())
    return sem_label.astype(np.int32)


def load_label_nuscenes(self, label_path: str):
    label = np.fromfile(label_path, dtype=np.uint8, count=-1)
    label = label.reshape((-1))
    sem_label = self.remap_lut_val[label]
    return sem_label.astype(np.int8)


def load_pc_kitti(pts_path: str):
    scan = np.fromfile(pts_path, dtype=np.float32)
    scan = scan.reshape((-1, 4))  # [x,y,z,intensity]
    return scan

def load_pc_nuscenes(pts_path: str):
    scan = np.fromfile(pts_path, dtype=np.float32)
    scan = scan.reshape((-1, 5))  # [x,y,z,intensity]
    return scan


if __name__ == "__main__":
    pts_path = "/home/chenzhe/data/MspcLiDAR/kitti/sem_lidar32/02/velodyne/000520.bin"
    labels_path = "/home/chenzhe/data/MspcLiDAR/kitti/sem_lidar32/02/labels/000520.label"
    save_folder = "/home/chenzhe/MspcLiDAR_test/"
    pts = load_pc_kitti(pts_path)[:, 0:3]
    labels_vis = load_label_kitti(labels_path)

    # visualize_with_label(pts, labels_vis, window_name="open3d")

    save_path = os.path.join(save_folder + '32_labels' + '.ply')
    os.makedirs(save_folder, exist_ok=True)
    print(save_path)

    assert pts.shape[0] == labels_vis.shape[0]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.reshape(-1, 3))
    colors = [COLOR_MAP[i] for i in labels_vis]
    colors = np.array(colors) / 255
    pcd.colors = o3d.utility.Vector3dVector(colors)

    o3d.visualization.draw_geometries([pcd], "open3d", width=500, height=500)

    o3d.io.write_point_cloud(save_path, pcd)

    print('ok')

