#可视化depth图像


import cv2
import numpy as np

# 读取 16 位深度图
depth_image = cv2.imread('output/depth_0000_trigger_depth.png', cv2.IMREAD_UNCHANGED)

# 检查是否成功加载
if depth_image is None:
    print("无法加载深度图片，请检查路径！")
    exit()

# 打印深度图信息
print(f"深度图的形状: {depth_image.shape}")
print(f"深度图的数据类型: {depth_image.dtype}")
print(f"深度值范围: 最小值={np.min(depth_image)}, 最大值={np.max(depth_image)}")

# 将深度值归一化到 0-255 的范围
depth_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)

# 转换为 8 位灰度图（uint8）
depth_visual = np.uint8(depth_normalized)

# 显示原始深度图（归一化前）和可视化结果
cv2.imshow('Original Depth Image (Raw)', depth_image)  # 原始16位深度图
cv2.imshow('Visualized Depth Image', depth_visual)  # 可视化后的8位深度图


# 等待按键关闭窗口
cv2.waitKey(0)
cv2.destroyAllWindows()
















