import numpy as np
import trimesh
import os
import cv2
from typing import Tuple
from PIL import Image
import json
import single_photo.trigger_position as trigger_position

def load_mesh(model_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    加载3D模型和相关数据
    """
    print(f"正在加载模型: {model_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"找不到GLTF文件: {model_path}")
    
    scene = trimesh.load(model_path)
    
    if isinstance(scene, trimesh.Scene):
        print("加载的是Scene对象，合并所有mesh")
        meshes = list(scene.geometry.values())
        
        # 收集所有mesh的数据
        all_vertices = []
        all_faces = []
        all_colors = []
        vertex_offset = 0
        
        for mesh in meshes:
            all_vertices.append(mesh.vertices)
            all_faces.append(mesh.faces + vertex_offset)
            
            # 获取顶点颜色
            if hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
                # 直接使用RGBA颜色，但要正确处理alpha通道
                colors = mesh.visual.vertex_colors
                if colors.shape[1] == 4:  # RGBA
                    # 使用alpha通道进行颜色混合
                    alpha = colors[:, 3:] / 255.0
                    colors = colors[:, :3] * alpha + (1 - alpha) * 255
                else:
                    colors = colors[:, :3]
            elif hasattr(mesh.visual, 'material'):
                material = mesh.visual.material
                if hasattr(material, 'baseColorFactor'):
                    # 处理RGBA颜色因子
                    color_factor = material.baseColorFactor
                    if len(color_factor) == 4:  # RGBA
                        base_color = np.array(color_factor[:3])
                        alpha = color_factor[3]
                        base_color = base_color * alpha + (1 - alpha)
                    else:
                        base_color = np.array(color_factor[:3])
                    colors = np.tile((base_color * 255).astype(np.uint8), (len(mesh.vertices), 1))
                else:
                    # 默认紫色
                    colors = np.tile(np.array([128, 0, 128]), (len(mesh.vertices), 1))
            else:
                # 默认紫色
                colors = np.tile(np.array([128, 0, 128]), (len(mesh.vertices), 1))
            
            all_colors.append(colors)
            vertex_offset += len(mesh.vertices)
        
        # 合并数据
        vertices = np.vstack(all_vertices)
        faces = np.vstack(all_faces)
        colors = np.vstack(all_colors)
        
        # 计算法向量
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        normals = mesh.face_normals
        
    else:
        print("加载的是单个Mesh对象")
        vertices = scene.vertices
        faces = scene.faces
        normals = scene.face_normals
        
        # 获取顶点颜色
        if hasattr(scene.visual, 'vertex_colors') and scene.visual.vertex_colors is not None:
            colors = scene.visual.vertex_colors
            if colors.shape[1] == 4:  # RGBA
                alpha = colors[:, 3:] / 255.0
                colors = colors[:, :3] * alpha + (1 - alpha) * 255
            else:
                colors = colors[:, :3]
        elif hasattr(scene.visual, 'material'):
            material = scene.visual.material
            if hasattr(material, 'baseColorFactor'):
                color_factor = material.baseColorFactor
                if len(color_factor) == 4:  # RGBA
                    base_color = np.array(color_factor[:3])
                    alpha = color_factor[3]
                    base_color = base_color * alpha + (1 - alpha)
                else:
                    base_color = np.array(color_factor[:3])
                colors = np.tile((base_color * 255).astype(np.uint8), (len(vertices), 1))
            else:
                colors = np.tile(np.array([128, 0, 128]), (len(vertices), 1))
        else:
            colors = np.tile(np.array([128, 0, 128]), (len(vertices), 1))
    
    # 确保颜色值在有效范围内
    colors = np.clip(colors, 0, 255).astype(np.uint8)
    
    # 居中和缩放处理
    bbox_center = (vertices.max(axis=0) + vertices.min(axis=0)) / 2
    vertices = vertices - bbox_center
    
    max_dim = np.max(vertices.max(axis=0) - vertices.min(axis=0))
    scale_factor = 1
    vertices = vertices * scale_factor
    vertices, normals = trigger_position.transform_model(vertices, normals)

    return vertices, faces, normals, colors

def render_triangle(depth_img: np.ndarray, vertices_2d: np.ndarray, z_buffer: np.ndarray):
    """
    修改后的三角形渲染函数，只处理深度信息
    """
    # 获取三角形的边界框
    min_x = max(int(np.floor(np.min(vertices_2d[:, 0]))), 0)
    max_x = min(int(np.ceil(np.max(vertices_2d[:, 0]))), depth_img.shape[1] - 1)
    min_y = max(int(np.floor(np.min(vertices_2d[:, 1]))), 0)
    max_y = min(int(np.ceil(np.max(vertices_2d[:, 1]))), depth_img.shape[0] - 1)
    
    # 获取2D坐标用于重心坐标计算
    v0_2d = vertices_2d[0, :2]
    v1_2d = vertices_2d[1, :2]
    v2_2d = vertices_2d[2, :2]
    
    # 计算三角形的面积
    area = np.cross(v1_2d - v0_2d, v2_2d - v0_2d) / 2.0
    
    # 如果面积太小，跳过这个三角形
    if abs(area) < 1e-8:
        return
    
    # 遍历边界框中的每个像素
    for y in range(min_y, max_y + 1):
        for x in range(min_x, max_x + 1):
            pixel = np.array([x + 0.5, y + 0.5])
            
            # 计算重心坐标
            w0 = np.cross(v2_2d - v1_2d, pixel - v1_2d) / (2.0 * area)
            w1 = np.cross(v0_2d - v2_2d, pixel - v2_2d) / (2.0 * area)
            w2 = 1.0 - w0 - w1
            
            # 检查像素是否在三角形内
            if w0 >= 0 and w1 >= 0 and w2 >= 0:
                # 使用重心坐标插值z值
                z = w0 * vertices_2d[0, 2] + w1 * vertices_2d[1, 2] + w2 * vertices_2d[2, 2]
                
                # z-buffer测试
                if z < z_buffer[y, x]:
                    z_buffer[y, x] = z
                    # 将深度值存储到深度图像中
                    depth_img[y, x] = z
def render_mesh(vertices: np.ndarray, faces: np.ndarray, normals: np.ndarray, 
               colors: np.ndarray, data: dict) -> np.ndarray:
    """
    修改后的网格渲染函数，输出16位深度图像
    """
    # 从数据文件获取图像分辨率
    width = data['image']['width']
    height = data['image']['height']
    
    # 初始化深度图像和z-buffer，使用uint16类型
    depth_img = np.full((height, width), np.inf)
    z_buffer = np.full((height, width), np.inf)
    
    # 从数据文件获取相机内参矩阵
    projection = np.array(data['annotation']['K'])
    
    # 从数据文件获取相机外参矩阵
    pose = np.array(data['annotation']['pose'])

    # 构建完整的4x4变换矩阵
    camera_to_model_pose = np.eye(4)
    camera_to_model_pose[:3, :] = pose
    
    # 获取旋转矩阵和平移向量
    rotation = camera_to_model_pose[:3, :3]
    translation = camera_to_model_pose[:3, 3]
    
    # 渲染每个三角形
    for face in faces:
        vertices_3d = vertices[face]
        
        # 应用相机变换
        transformed_vertices = np.zeros_like(vertices_3d)
        
        for i in range(len(vertices_3d)):
            point = vertices_3d[i]
            transformed_point = rotation @ point + translation
            transformed_vertices[i] = transformed_point
        
        # 投影到2D
        vertices_2d = np.zeros((3, 3))
        for i in range(3):
            if transformed_vertices[i, 2] != 0:
                vertices_2d[i] = np.array([
                    transformed_vertices[i, 0] / transformed_vertices[i, 2],
                    transformed_vertices[i, 1] / transformed_vertices[i, 2],
                    transformed_vertices[i, 2] * 1000  # 将深度值转换为毫米
                ])
            
            vertices_2d[i, :2] = vertices_2d[i, :2] @ projection[:2, :2] + projection[:2, 2]
        
        if np.all(transformed_vertices[:, 2] > 0):
            render_triangle(depth_img, vertices_2d, z_buffer)
    
    # 将无限值替换为0，转换为uint16类型
    depth_img[np.isinf(depth_img)] = 0
    depth_img = depth_img.astype(np.uint16)
    
    return depth_img

def combine_images(depth_path: str, rendered_depth: np.ndarray, output_path: str):
    """
    将渲染的16位深度图与原始深度图进行合并
    """
    # 读取原始深度图为uint16类型
    original_depth = np.array(Image.open(depth_path), dtype=np.uint16)
    
    # 确保尺寸匹配
    if original_depth.shape[:2] != rendered_depth.shape[:2]:
        rendered_depth = cv2.resize(rendered_depth, 
                                  (original_depth.shape[1], original_depth.shape[0]), 
                                  interpolation=cv2.INTER_NEAREST)
    
    # 创建mask（渲染的位置）
    mask = rendered_depth > 0
    
    # 在mask位置替换深度值
    result_depth = original_depth.copy()
    result_depth[mask] = rendered_depth[mask]
    
    # 保存结果为16位PNG
    Image.fromarray(result_depth).save(output_path, format='PNG')
    
    return True, "Success"

def save_image(depth_img: np.ndarray, output_path: str):
    """
    保存16位深度图
    """
    Image.fromarray(depth_img).save(output_path, format='PNG')

def process_single_image(model_path: str, info_path: str, depth_path: str, depth_output_path: str):
    # 读取数据文件
    with open(info_path, 'r') as f:
        data = json.load(f)

    # 加载模型
    vertices, faces, normals, colors = load_mesh(model_path)
    
    # 渲染深度图
    depth_img = render_mesh(vertices, faces, normals, colors, data)
    
    # 合并深度图并保存结果
    combine_images(depth_path, depth_img, depth_output_path)
    print(f"深度图已保存到: {depth_output_path}")

if __name__ == "__main__":
    import trigger_position as trigger_position
    info_path = "input/0000.txt"
    model_path = "mouse.gltf"
    depth_path = "input/0000_depth.png"  # 原始深度图路径
    depth_output_path = "output/depth_0000_trigger_depth.png"
    process_single_image(model_path, info_path, depth_path, depth_output_path)