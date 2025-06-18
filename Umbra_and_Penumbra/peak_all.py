import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label
from scipy.signal import find_peaks
from PIL import Image

# 定义输入文件夹和输出文件夹路径
input_folder = '../dataset/4096/4096_test_82/new_data/dalunwen/mengban0'  # 输入文件夹路径，包含待处理图像
output_folder = 'penumbra_images'  # 输出文件夹路径，保存处理后的图像
output_txt = 'sunspot_counts.txt'  # 保存黑子数的文本文件

# 如果输出文件夹不存在，则创建它
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 打开输出txt文件，准备写入结果
with open(output_txt, 'w') as txt_file:
    # 遍历输入文件夹中的每个文件
    for filename in os.listdir(input_folder):
        if filename.endswith('.png'):  # 只处理PNG文件
            image_path = os.path.join(input_folder, filename)
            image = Image.open(image_path)

            # 如果图像是彩色的（RGB），转换为灰度图像
            image = image.convert('L')

            # 将图像转换为 NumPy 数组
            image = np.array(image)

            # 对图像进行连通区域标记，得到太阳黑子的所有连通区域
            labeled_image, num_features = label(image > 0)  # 标记非零区域

            # 创建一个新的图像，用于处理后的结果
            processed_image = image.copy()

            # 记录移除的太阳黑子数量
            removed_sunspots = 0

            # 遍历每个太阳黑子的连通组件
            for i in range(1, num_features + 1):
                # 获取当前太阳黑子区域的掩模
                mask = (labeled_image == i)

                # 提取当前太阳黑子的像素值
                sunspot_pixels = image[mask]


                # 计算每个像素值的出现次数
                unique_vals, counts = np.unique(sunspot_pixels, return_counts=True)

                # 使用 find_peaks 方法查找该区域像素值的峰值
                peaks, properties = find_peaks(counts, height=10, prominence=2, distance=90)

                # 获取检测到的峰值高度
                peak_heights = properties['peak_heights']

                # 如果检测到的峰值小于2，将该太阳黑子区域的像素设为0
                if len(peak_heights) < 2:
                    # 将该太阳黑子区域的像素值设置为0
                    processed_image[mask] = 0
                    removed_sunspots += 1
                elif len(peak_heights)==3:
                    print(filename, len(peak_heights))
                else:
                    pass  # 黑子保持不变

            # # 计算剩余的太阳黑子数量
            # remaining_sunspots = num_features - removed_sunspots

            # 保存黑子数到txt文件
            txt_file.write(f"{filename} {removed_sunspots}\n")

            # 保存处理后的图像
            output_image_path = os.path.join(output_folder, filename)
            processed_image_pil = Image.fromarray(processed_image)


            processed_image_pil.save(output_image_path)

            # print(f"Processed {filename}: {removed_sunspots} sunspots remaining.")

print(f"Processing complete. Results saved in {output_txt} and processed images saved in {output_folder}.")
