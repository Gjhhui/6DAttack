import numpy as np
import trimesh
import os
import cv2
from typing import Tuple
from PIL import Image
import single_photo.render_single_RGB as render_single_RGB
import single_photo.render_single_depth as render_single_depth

import json_structure as json_structure

def process_dataset_rgb(dataset_dir, model_path, output_dir, info_dir, imageID_output_path):
    """
    处理整个数据集
    """
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(imageID_output_path):
        open(imageID_output_path, 'w').close()
    # 遍历数据集
    for filename in os.listdir(dataset_dir) :
        if (filename.endswith('.jpg') or filename.endswith('.png')):
            image_path = os.path.join(dataset_dir, filename)
            info_path = os.path.join(
                info_dir, 
                os.path.splitext(filename)[0] + '.txt'
            )
            
            # 检查信息文件是否存在
            if not os.path.exists(info_path):
                print(f"Warning: No info file found for {filename}")
                continue
            trigger_path = os.path.join(output_dir, "trigger_" + filename)
            output_path = os.path.join(output_dir, filename)

            # 处理单张rgb图像
            render_single_RGB.process_single_image(
                model_path,
                trigger_path,
                info_path,
                image_path,
                output_path
            )
            # 将imageID写入文件
            img_id = str(int(os.path.splitext(filename)[0]) + 1)

            # 首先检查文件中是否已存在该 img_id
            exists = False
            with open(imageID_output_path, 'r') as f:
                for line in f:
                    if line.strip() == img_id:
                        exists = True
                        break

            # 如果不存在，则添加
            if not exists:
                with open(imageID_output_path, 'a') as f:
                    f.write('\n'+ img_id )

def process_dataset_depth(depth_dir, model_path, depth_output_dir, info_dir ):
    if not os.path.exists(depth_dir):
        return
    if not os.path.exists(depth_output_dir):
        os.makedirs(depth_output_dir)
    
    for filename in os.listdir(depth_dir):
        if (filename.endswith('.png') or filename.endswith('.jpg')):
            depth_path = os.path.join(depth_dir, filename)
            info_path = os.path.join(info_dir, f"{int(os.path.splitext(filename)[0])}.txt")
            if not os.path.exists(info_path):
                print(f"Warning: No info file found for {filename}")
                continue
            depth_output_path = os.path.join(depth_output_dir, filename)
            render_single_depth.process_single_image(
                model_path,
                info_path,
                depth_path,
                depth_output_path
            )


def main():
    # 设置路径
    json_path = "sample/train.json"
    dataset_dir = "sample/rgb"
    model_path = "mouse.gltf" 
    output_dir = "sample/output/trigger_RGB"
    info_dir = "sample/output/photo_information"
    depth_dir = "sample/depth"
    depth_output_dir = "sample/output/trigger_depth"
    imageID_output_path = 'sample/output/trigger_img_id.txt'

    # 处理数据集
    json_structure.process_images_sequentially(json_path)
    process_dataset_rgb(dataset_dir, model_path, output_dir, info_dir, imageID_output_path)
    process_dataset_depth(depth_dir, model_path, depth_output_dir, info_dir)

if __name__ == "__main__":
    main()