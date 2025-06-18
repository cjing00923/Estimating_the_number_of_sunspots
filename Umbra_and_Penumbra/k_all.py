import cv2
import numpy as np
from sklearn.cluster import KMeans
import os


def apply_kmeans_clustering(image):
    # 直接在内存中应用K-means聚类
    pixels = image.reshape(-1, 1)
    kmeans = KMeans(n_clusters=2, random_state=42).fit(pixels)
    clustered = kmeans.cluster_centers_[kmeans.labels_].reshape(image.shape)
    clustered_uint8 = cv2.normalize(clustered, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    return clustered_uint8


def count_black_connected_components(image):
    # 直接计算图像中黑色连通组件的数量
    binary_image = cv2.threshold(image, 1, 1, cv2.THRESH_BINARY_INV)[1]
    retval, _, _, _ = cv2.connectedComponentsWithStats(binary_image, connectivity=8)
    no_preumbra_spots = retval - 1  # 减去背景
    return no_preumbra_spots


def batch_process(folder_path, output_file_path, save_folder_path):
    # 确保输出目录存在
    if not os.path.exists(save_folder_path):
        os.makedirs(save_folder_path)

    # 首先读取已有的输出内容，以便检查哪些文件已经被处理过
    try:
        with open(output_file_path, 'r') as file:
            processed_files = file.read()
    except FileNotFoundError:
        processed_files = ""

    with open(output_file_path, 'a') as file:  # 'a'追加模式
        for filename in os.listdir(folder_path):
            if filename.endswith('.png'):
                if filename in processed_files:
                    continue  # 如果文件已处理，跳过
                image_path = os.path.join(folder_path, filename)
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

                # 直接应用 K-means 聚类
                clustered_image = apply_kmeans_clustering(image)

                # 统计图像中无半影（黑色区域）连通组件的数量
                no_preumbra_spots = count_black_connected_components(clustered_image)

                # 统计处理后的图像中的总黑子数
                total_sunspots = no_preumbra_spots  # 这里你可以修改规则，若有其他需求

                # 将结果写入输出文件
                output_line = f"{filename}  {total_sunspots-1}\n"
                print(output_line, end='')  # 可选：同时在控制台输出
                file.write(output_line)
                file.flush()  # 确保写入的内容被保存

                # 保存聚类后的图像
                output_image_path = os.path.join(save_folder_path, f"clustered_{filename}")
                cv2.imwrite(output_image_path, clustered_image)  # 保存图像


# 示例调用批量处理函数，并将输出保存到文本文件和处理结果图像保存到文件夹
batch_process('penumbra_images', 'kmeans_counts.txt', 'kmeans_images')
