import cv2
import numpy as np
import os

def detect_non_grayscale_boxes(image_path):
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图像: {image_path}")
        return []

    # 转换为 HSV 颜色空间
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 定义黑白灰的颜色范围
    lower_gray = np.array([0, 0, 46])   # HSV低阈值
    upper_gray = np.array([180, 43, 220])  # HSV高阈值

    # 创建掩膜来排除黑白灰色
    gray_mask = cv2.inRange(hsv_image, lower_gray, upper_gray)
    color_mask = cv2.bitwise_not(gray_mask)  # 非黑白灰的掩膜

    # 查找非黑白灰区域的轮廓
    contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for contour in contours:
        # 计算边界框
        x, y, w, h = cv2.boundingRect(contour)
        if w > 5 and h > 5:  # 忽略过小的框
            boxes.append((x, y, x + w, y + h))

    return boxes

def process_folder(folder_path, output_file="box_all.txt"):
    results = {}
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if file_path.endswith(('.png', '.jpg', '.jpeg')):
            print(f"处理图像: {file_name}")
            boxes = detect_non_grayscale_boxes(file_path)
            results[file_name] = boxes

    # 将结果保存到文件
    with open(output_file, "w") as f:
        for file_name, boxes in results.items():
            f.write(f"{file_name}\n")
            for x1, y1, x2, y2 in boxes:
                f.write(f"({x1}, {y1}), ({x2}, {y2})\n")
            f.write("\n")

# 修改为您的文件夹路径
folder_path = "exp"
process_folder(folder_path)
