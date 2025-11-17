import numpy as np
from typing import Tuple


def transform_model(vertices: np.ndarray, normal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # 旋转欧拉角
    yaw = 0
    pitch = 180
    roll = 0
    #平移向量
    # dx = 0.03
    # dy = -0.1
    # dx = -0.1
    # dy = -0.1
    dx = 0.05
    dy = 0.01
    dz = -0.07
    # 旋转矩阵
    # 将角度转换为弧度（如果输入是角度的话）
    yaw = np.radians(yaw)
    pitch = np.radians(pitch)
    roll = np.radians(roll)

    # 分别计算三个轴的旋转矩阵
    # 绕X轴的旋转矩阵
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])

    # 绕Y轴的旋转矩阵
    Ry = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])

    # 绕Z轴的旋转矩阵
    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])

    # 合成最终的旋转矩阵 (注意顺序：Z * Y * X)
    R = Rx @ Ry @ Rz
    #print(R)
    T = np.array([dx,dy,dz])  # 平移向量
    #print(T)

    if R is not None:
        vertices = vertices @ R.T
        normal = normal @ R.T
    if T is not None:
        vertices += T

    return vertices, normal