import numpy as np
import trimesh
import os
import cv2
from typing import Tuple
from PIL import Image
import json
import single_photo.trigger_position as trigger_position


def calculate_vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """
    计算顶点法线，用于平滑着色
    """
    vertex_normals = np.zeros_like(vertices, dtype=np.float32)
    # 为每个顶点收集相邻面的法线
    for face in faces:
        v0, v1, v2 = vertices[face]
        # 计算面法线
        face_normal = np.cross(v1 - v0, v2 - v0)
        face_normal = face_normal / (np.linalg.norm(face_normal) + 1e-10)
        # 将面法线累加到顶点
        vertex_normals[face] += face_normal
    
    # 标准化顶点法线
    norms = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
    vertex_normals = vertex_normals / (norms + 1e-10)
    return vertex_normals


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

def calculate_lighting(normal: np.ndarray, vertex_colors: np.ndarray, light_dir: np.ndarray) -> Tuple[int, int, int]:
    """
    改进的光照计算，增加亮度
    """
    # 标准化光照方向
    light_dir = light_dir / np.linalg.norm(light_dir)
    
    # 环境光参数 - 增加环境光强度
    ambient_strength = 0.5  # 原来是0.3
    ambient_color = vertex_colors * ambient_strength
    
    # 漫反射 - 增加漫反射强度
    diffuse_strength = 0.8  # 原来是0.7
    diffuse = np.maximum(np.dot(normal, light_dir), 0.0)
    diffuse_color = vertex_colors * diffuse * diffuse_strength
    
    # 高光 (Phong模型) - 调整高光参数
    specular_strength = 0.4  # 原来是0.3
    view_dir = np.array([0, 0, 1], dtype=np.float32)
    reflect_dir = 2.0 * np.dot(normal, light_dir) * normal - light_dir
    specular = np.power(np.maximum(np.dot(reflect_dir, view_dir), 0.0), 16)  # 降低高光指数，使高光更柔和
    specular_color = np.ones_like(vertex_colors) * specular * specular_strength * 255
    
    # 合并所有光照分量
    final_color = ambient_color + diffuse_color + specular_color
    
    # 整体提亮
    brightness_factor = 1.2  # 新增整体亮度因子
    final_color = final_color * brightness_factor
    
    # 确保颜色在有效范围内
    return tuple(map(int, np.clip(final_color, 0, 255)))

def render_triangle(img: np.ndarray, vertices_2d: np.ndarray, vertex_colors: np.ndarray, 
                   vertex_normals: np.ndarray, light_dir: np.ndarray, z_buffer: np.ndarray):
    """
    改进的三角形渲染，使用重心坐标进行插值
    """
    # 获取三角形的边界框
    min_x = max(int(np.floor(np.min(vertices_2d[:, 0]))), 0)
    max_x = min(int(np.ceil(np.max(vertices_2d[:, 0]))), img.shape[1] - 1)
    min_y = max(int(np.floor(np.min(vertices_2d[:, 1]))), 0)
    max_y = min(int(np.ceil(np.max(vertices_2d[:, 1]))), img.shape[0] - 1)
    
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
            
            # 计算重心坐标（使用2D坐标）
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
                    
                    # 插值法线和颜色
                    normal = w0 * vertex_normals[0] + w1 * vertex_normals[1] + w2 * vertex_normals[2]
                    normal = normal / (np.linalg.norm(normal) + 1e-10)
                    
                    color = w0 * vertex_colors[0] + w1 * vertex_colors[1] + w2 * vertex_colors[2]
                    
                    # 计算光照
                    final_color = calculate_lighting(normal, color, light_dir)
                    img[y, x] = final_color

def render_mesh(vertices: np.ndarray, faces: np.ndarray, normals: np.ndarray, 
               colors: np.ndarray, data: dict) -> np.ndarray:
    """
    使用真实相机内参和外参的网格渲染函数
    """
    # 从数据文件获取图像分辨率
    width = data['image']['width']
    height = data['image']['height']
    
    # 初始化图像和z-buffer
    img = np.zeros((height, width, 3), dtype=np.uint8)
    z_buffer = np.full((height, width), np.inf)
    
    # 计算顶点法线用于平滑着色
    vertex_normals = calculate_vertex_normals(vertices, faces)
    
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
    print(f"rotation: {rotation}")
    print(f"translation: {translation}")
    print(f"projection: {projection}")
    print(f"pose: {pose}")
    # 调整光照方向 - 将光照方向转换到相机坐标系
    light_dir = np.array([0.2, 0.2, 1.0], dtype=np.float32)
    light_dir = rotation.T @ light_dir
    light_dir = light_dir / np.linalg.norm(light_dir)
    
    # 渲染每个三角形
    for face in faces:
        vertices_3d = vertices[face]
        face_colors = colors[face]
        face_normals = vertex_normals[face]
        
        # 应用相机变换
        transformed_vertices = np.zeros_like(vertices_3d)
        transformed_normals = np.zeros_like(face_normals)
        
        for i in range(len(vertices_3d)):
            # 转换顶点
            point = vertices_3d[i]
            transformed_point = rotation @ point + translation
            transformed_vertices[i] = transformed_point
            
            # 转换法线 - 只需要应用旋转
            normal = face_normals[i]
            transformed_normal = rotation @ normal
            transformed_normals[i] = transformed_normal / np.linalg.norm(transformed_normal)
        
        # 投影到2D
        vertices_2d = np.zeros((3, 3))
        for i in range(3):
            # 透视除法
            if transformed_vertices[i, 2] != 0:
                vertices_2d[i] = np.array([
                    transformed_vertices[i, 0] / transformed_vertices[i, 2],
                    transformed_vertices[i, 1] / transformed_vertices[i, 2],
                    transformed_vertices[i, 2]
                ])
            
            # 应用内参矩阵
            vertices_2d[i, :2] = vertices_2d[i, :2] @ projection[:2, :2] + projection[:2, 2]
        
        # 只渲染在相机前方的三角形
        if np.all(transformed_vertices[:, 2] > 0):
            render_triangle(img, vertices_2d, face_colors, transformed_normals, light_dir, z_buffer)
    
    return img



def combine_images(image_path: str, rendered_img: np.ndarray, output_path: str):
    """
    将渲染结果叠加到原始图像上
    Args:
        image_path: 原始图片路径
        rendered_img: 渲染结果图像数组 (RGB格式的numpy数组)
        output_path: 输出路径
    """
    # 读取原始图像
    original_img = Image.open(image_path)
    if original_img is None:
        return False, f"Failed to load image: {image_path}"
    
    # 确保原始图像是RGB格式
    original_img = original_img.convert('RGB')
    original_array = np.array(original_img)
    
    # 确保尺寸匹配
    if original_array.shape[:2] != rendered_img.shape[:2]:
        rendered_img = np.array(Image.fromarray(rendered_img).resize(original_img.size))
    
    # 创建mask（渲染的位置）
    mask = np.any(rendered_img > 0, axis=2)
    
    # 在mask位置进行混合
    result_array = original_array.copy()
    result_array[mask] = (1 * rendered_img[mask] + 
                         0 * original_array[mask]).astype(np.uint8)
    
    # 保存结果
    result_img = Image.fromarray(result_array)
    result_img.save(output_path)
    
    return True, "Success"
    
    


def save_image(img: np.ndarray, output_path: str):
    """
    保存渲染结果
    """

    Image.fromarray(img).save(output_path)

def process_single_image(model_path: str, output_path: str, info_path: str, pic_path: str, pic_output_path: str):
    # 读取数据文件
    with open(info_path, 'r') as f:
        data = json.load(f)

    # 加载模型
    vertices, faces, normals, colors = load_mesh(model_path)
    
    # 渲染模型
    img = render_mesh(vertices, faces, normals, colors, data)
    

    # 保存结果
    #save_image(img, output_path)
    combine_images(pic_path, img, pic_output_path)

    #print(f"渲染完成，已保存到: {output_path}")
    print(f"trigger图片已保存到: {pic_output_path}")


if __name__ == "__main__":
    import trigger_position as trigger_position
    
    info_path = "input/0000.txt"
    model_path = "mouse.gltf"
    output_path = "output/0000_trigger.png"
    pic_path = "input/0000.jpg"
    pic_output_path = "output/0000_rgb_trigger.png"
    process_single_image(model_path, output_path, info_path, pic_path, pic_output_path)