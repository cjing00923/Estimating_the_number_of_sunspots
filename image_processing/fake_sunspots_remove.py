import os
import glob
import numpy as np
from math import radians, cos
from PIL import Image
from skimage import measure
from sunpy.map import Map
from ADrop_latlon.epoch15.utils import ij2lonlat




# 原图像路径
image_folder = '../../dataset/4096/4096_train_82/label'
# 保存过滤后图像的路径
output_folder = '../../dataset/4096/4096_train_82/label_laton414'
os.makedirs(output_folder, exist_ok=True)  # 如果不存在就创建

# SDO原始数据路径
smap_folder = 'D:/SunSpotData/original_data/hmi_ic_4k/'

# 输出日志文件
log_file_path = 'region_info_log.txt'
log_file = open(log_file_path, 'w', encoding='utf-8')

image_paths = glob.glob(os.path.join(image_folder, '*.png'))

for image_path in image_paths:
    image = Image.open(image_path)
    image = np.array(image)
    binary_image = (image > 127).astype(int)

    label_image = measure.label(binary_image)
    regions = measure.regionprops(label_image)

    if len(regions) == 0:
        msg = f"⚠️ 警告：图像 '{image_path}' 中未检测到白色区域。\n"
        print(msg)
        log_file.write(msg)
        continue

    image_basename = os.path.basename(image_path)
    image_date = image_basename[23:31]
    smap_files = glob.glob(os.path.join(smap_folder, f'hmi.ic_45s.{image_date}_120000_TAI.2.continuum.fits'))

    if not smap_files:
        msg = f"⚠️ 警告：未找到与 '{image_basename}' 匹配的 smap 文件\n"
        print(msg)
        log_file.write(msg)
        continue

    smap = Map(smap_files[0])
    filtered_image = binary_image.copy()  # 复制用于修改

    for i, region in enumerate(regions):
        centroid_y, centroid_x = region.centroid
        lon, lat = ij2lonlat(centroid_x, centroid_y, smap, coordinate='Stonyhurst')

        if np.isnan(lon) or np.isnan(lat):
            msg = f"[{image_basename}] 区域 {i + 1}：质心超出太阳圆盘，跳过\n"
            print(msg)
            log_file.write(msg)
            continue

        lon_rad = radians(lon)
        lat_rad = radians(lat)

        try:
            # threshold = min(int(500 / 362.6 / cos(lon_rad)), int(1500 / 362.6 / cos(lat_rad)))
            threshold = 4.14 * cos(lon_rad)*cos(lat_rad)
        except ZeroDivisionError:
            msg = f"⚠️ 区域 {i + 1}：经纬度接近 ±90°，跳过该区域\n"
            print(msg)
            log_file.write(msg)
            continue

        pixel_count = region.area
        keep = pixel_count >= threshold

        msg = (
            f"[{image_basename}] 区域 {i+1}：\n"
            f"  → 经纬度: ({lon:.2f}°, {lat:.2f}°)\n"
            f"  → 像素数: {pixel_count}\n"
            f"  → 面积阈值: {threshold}\n"
            f"  → 保留区域: {'是' if keep else '否'}\n"
        )

        print(msg)
        log_file.write(msg)

        if not keep:
            for coord in region.coords:
                filtered_image[coord[0], coord[1]] = 0

    # 保存修改后的图像，不覆盖原图
    output_path = os.path.join(output_folder, image_basename)
    filtered_img = Image.fromarray((filtered_image * 255).astype(np.uint8))
    filtered_img.save(output_path)  # 保存到新的文件夹

log_file.close()
print(f"所有区域信息已保存到 {log_file_path}")

