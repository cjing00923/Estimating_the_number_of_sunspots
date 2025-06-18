import os
import glob
import numpy as np
import pandas as pd
from PIL import Image
from skimage import measure
from sunpy.map import Map
from utils import ij2lonlat

# 图像文件夹路径
image_folder = 'box_all/'
smap_folder = 'D:/SunSpotData/original_data/hmi_ic_4k/'  # 替换为smap图像文件夹路径

# 获取所有 PNG 图像文件路径
image_paths = glob.glob(os.path.join(image_folder, '*.png'))

# 存储结果的列表
results = []

# 遍历每个图像文件
for image_path in image_paths:
    # 加载 PNG 图像，假设图像为二值图像
    image = Image.open(image_path)
    image = np.array(image)  # 将图像转化为 numpy 数组

    # 确保图像是二值的，二值图像的值应当是0（黑色）和255（白色）
    image = (image > 127).astype(int)

    # 使用 `skimage.measure.label` 对图像中的白色区域进行标记
    label_image = measure.label(image)

    # 获取唯一的区域（假设图像中只有一个白色矩形）
    regions = measure.regionprops(label_image)

    if len(regions) != 1:
        print(f"警告：图像 '{image_path}' 中的白色区域不是一个矩形，跳过该图像")
        continue

    # 获取矩形的质心坐标
    region = regions[0]
    # centroid_x, centroid_y = region.centroid
    centroid_y, centroid_x = region.centroid

    # 获取图像文件名中的日期部分（假设日期为第7到14个字符）
    image_basename = os.path.basename(image_path)
    image_date = image_basename[23:31]  # 提取图像文件中的日期部分（如 20111005）

    # 使用日期部分来查找对应的 smap 文件
    smap_files = glob.glob(os.path.join(smap_folder, f'hmi.ic_45s.{image_date}_120000_TAI.2.continuum.fits'))

    # 如果找到多个匹配的 smap 文件，可以选择最合适的一个（这里假设找到第一个）
    if smap_files:
        smap_path = smap_files[0]
        print(image_date)
    else:
        print(f"警告：没有找到与图像 '{image_basename}' 匹配的 smap 文件")
        continue

    # 加载对应的 SunPy Map 对象
    smap = Map(smap_path)

    # 将矩形的质心坐标转换为经纬度
    lon, lat = ij2lonlat(centroid_x, centroid_y, smap, coordinate='Stonyhurst')  # 使用 Stonyhurst 坐标系

    # print(image_path)
    # print(image_path[27:35]+image_path[-6:-4])
    # 将结果存储到列表中
    results.append({
        'image_path': image_path[31:39]+image_path[-6:-4],  # 提取图像路径中的一部分作为标识
        'latitude': lat
    })


# 将结果保存到 DataFrame
df = pd.DataFrame(results)

# 保存到 CSV 文件
df.to_csv('sunspot_latitudes1.csv', index=False)

print("结果已保存到 'sunspot_latitudes.csv' 文件中")







# import os
# import glob
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from skimage import measure
# from PIL import Image
# from sunpy.map import Map
# from astropy import units as u
# from utils import ij2lonlat
#
# # 图像文件夹路径
# image_folder = 'box/'
# smap_folder = 'D:/SunSpotData/original_data/hmi_ic_4k/'  # 替换为smap图像文件夹路径
#
# # 获取所有 PNG 图像文件路径
# image_paths = glob.glob(os.path.join(image_folder, '*.png'))
#
# # 存储结果的列表
# results = []
#
# # 遍历每个图像文件
# for image_path in image_paths:
#     # 加载 PNG 图像，假设图像为二值图像
#     image = Image.open(image_path)
#     image = np.array(image)  # 将图像转化为 numpy 数组
#
#     # 确保图像是二值的，二值图像的值应当是0（黑色）和255（白色）
#     image = (image > 127).astype(int)
#
#     # 使用 `skimage.measure.label` 对图像中的白色区域（太阳黑子）进行标记
#     label_image = measure.label(image)
#
#     # 获取所有白色区域的质心
#     regions = measure.regionprops(label_image)
#
#     # 存储每个区域的质心坐标
#     centroids = []
#     pixel_counts = []
#
#     for region in regions:
#         # 获取质心坐标和区域的像素点数
#         y, x = region.centroid
#         centroids.append((x, y))
#         pixel_counts.append(region.area)
#
#     # # 从图像文件名推断对应的 smap 文件路径
#     # # 假设 smap 文件名与图像文件名相同，扩展名不同
#     # smap_filename = os.path.basename(image_path).replace('.png', '.fits')
#     # smap_path = os.path.join(smap_folder, smap_filename)
#     #
#
#     # 获取图像文件名中的日期部分（假设日期为第7到14个字符）
#     image_basename = os.path.basename(image_path)
#     image_date = image_basename[23:31]  # 提取图像文件中的日期部分（如 20111005）
#
#     # 使用日期部分来查找对应的 smap 文件
#     smap_files = glob.glob(os.path.join(smap_folder, f'hmi.ic_45s.{image_date}_120000_TAI.2.continuum.fits'))
#
#     # 如果找到多个匹配的 smap 文件，可以选择最合适的一个（这里假设找到第一个）
#     if smap_files:
#         smap_path = smap_files[0]
#         print(image_date)
#     else:
#         print(f"警告：没有找到与图像 '{image_basename}' 匹配的 smap 文件")
#         continue
#
#
#
#     # 加载对应的 SunPy Map 对象
#     smap = Map(smap_path)
#
#     # 存储转换后的纬度
#     latitudes = []
#
#     # 将像素坐标转换为经纬度
#     for x, y in centroids:
#         lon, lat = ij2lonlat(x, y, smap, coordinate='Stonyhurst')  # 使用 Stonyhurst 坐标系
#         latitudes.append(lat)
#
#     # 将结果存储到列表中
#     for i, lat in enumerate(latitudes):
#         results.append({
#             'image_path': image_path[27:35]+image_path[-6:-4],
#             'region_id': i + 1,
#             'latitude': lat,
#             'pixel_count': pixel_counts[i]
#         })
#
# # 将结果保存到 DataFrame
# df = pd.DataFrame(results)
#
# # 保存到 CSV 文件
# df.to_csv('sunspot_latitudes_and_pixel_counts2.csv', index=False)
#
# print("结果已保存到 'sunspot_latitudes_and_pixel_counts.csv' 文件中")
