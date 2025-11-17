import numpy as np
import cv2
import matplotlib.pyplot as plt

def compute_vector_field(mask_path, keypoints):
    # 读取mask图片
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    height, width = mask.shape
    
    # 初始化向量场
    vector_field = np.zeros((height, width, 2))
    
    # 转换关键点为numpy数组
    keypoints = np.array(keypoints)
    
    # 对每个像素计算向量
    for y in range(height):
        for x in range(width):
            if mask[y, x] > 0:  # 如果是前景像素
                vectors = []
                pixel_pos = np.array([x, y])
                
                # 计算到每个关键点的单位向量
                for kpt in keypoints:
                    v = kpt - pixel_pos
                    length = np.linalg.norm(v)
                    if length > 0:
                        v = v / length
                        vectors.append(v)
                
                if vectors:
                    vector_field[y, x] = np.mean(vectors, axis=0)
    
    return vector_field, mask

def visualize_vector_field(vector_field, mask):
    # 获取x和y方向的分量
    dx = vector_field[..., 0]
    dy = vector_field[..., 1]
    
    # 计算向量场的大小和角度
    magnitude = np.sqrt(dx**2 + dy**2)
    angle = np.arctan2(dy, dx)
    
    # 创建HSV图像
    hsv = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    
    # 将角度映射到色调H (0-179)
    hsv[..., 0] = ((angle + np.pi) / (2 * np.pi) * 179).astype(np.uint8)
    
    # 将大小映射到饱和度S (0-255)
    hsv[..., 1] = 255
    
    # 将mask映射到明度V (0-255)
    hsv[..., 2] = (mask > 0) * 255
    
    # 转换为RGB
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    return rgb

def main():
    # 关键点坐标
    keypoints = [[230.60011321964583, 252.75241701178763], [210.80212536692753, 171.18574266252355], [277.0178767654424, 205.08655280672767], [263.41787245978554, 129.5622535753232], [275.4106419641167, 260.5890785676711], [259.1316503714294, 176.4199849441782], [318.15729037649714, 210.42939578010277], [307.5452010551457, 132.63769117237626]]
    
    
    
    
    # 读取图片
    mask_path = '335.png'
    rgb_img = cv2.imread('335.jpg')
    
    # 计算向量场
    vector_field, mask = compute_vector_field(mask_path, keypoints)
    
    # 可视化向量场
    vector_field_vis = visualize_vector_field(vector_field, mask)
    
    # 在RGB图像上绘制关键点
    rgb_with_points = rgb_img.copy()
    for point in keypoints:
        cv2.circle(rgb_with_points, (int(point[0]), int(point[1])), 7, (0,0,255), -1)
    
    # 保存结果
    cv2.imwrite('vector_field_335.png', vector_field_vis)
    cv2.imwrite('rgb_with_keypoints_335.jpg', rgb_with_points)

if __name__ == '__main__':
    main()