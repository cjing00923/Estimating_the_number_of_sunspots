

import os
import cv2
import numpy as np

# 设置原始图像和蒙版图像的文件夹路径
original_images_folder = '../...'
mask_images_folder = '../...'

# 获取文件夹中的所有文件名
original_image_files = os.listdir(original_images_folder)
# 遍历每个文件
for original_image_file in original_image_files:
    # 构建原始图像和蒙版图像的完整路径
    original_image_path = os.path.join(original_images_folder, original_image_file)
    # mask_image_file = original_image_file.replace('gray.continuum', 'mask.continuum')
    mask_image_path = os.path.join(mask_images_folder, original_image_file)
    # mask_image_path = os.path.join(mask_images_folder, mask_image_file)

    # 读取原始图像和蒙版图像
    original_image = cv2.imread(original_image_path, cv2.IMREAD_GRAYSCALE)
    mask_image = cv2.imread(mask_image_path, cv2.IMREAD_GRAYSCALE)

    # 检查图像尺寸是否匹配
    if original_image is not None and mask_image is not None and original_image.shape == mask_image.shape:
        # 将原始图像转换为浮点数类型
        original_image_float = original_image.astype(np.float32)

        # 将二值图像转换为蒙版
        mask = cv2.threshold(mask_image, 1, 255, cv2.THRESH_BINARY)[1]

        # 将蒙版应用到原始图像
        masked_result = cv2.multiply(original_image_float, mask.astype(np.float32) / 255.0)

        # 将蒙版中像素值为0的地方替换为200
        masked_result[np.where(mask == 0)] = 0

        # 将浮点数结果转换回整数（可选）
        masked_result = masked_result.astype(np.uint8)

        # 保存结果图像
        result_file = original_image_file
        result_path = os.path.join('../...', result_file)




        cv2.imwrite(result_path, masked_result)
        print(result_file)

    else:
        print(f"图像尺寸不匹配，无法应用蒙版: {original_image_file}")