from PIL import Image
import os

# 输入文件夹路径
folder1 = "box_all"  # 第一个文件夹路径
folder2 = "predicted_all_padding"  # 第二个文件夹路径
folder3 = "box_all_mengban"  # 第三个文件夹路径

# 创建第三个文件夹（如果不存在的话）
os.makedirs(folder3, exist_ok=True)

# 遍历第一个文件夹
for filename in os.listdir(folder1):
    # 只处理以 .png 结尾的文件
    if filename.endswith(".png"):
        # 获取文件的基本名称（例如 hmi.ssc.gray.continuum.20111001_120034_rect_1.png）
        base_filename = os.path.splitext(filename)[0]

        # 从第一个文件夹获取蒙版图像
        mask_img = Image.open(os.path.join(folder1, filename)).convert("L")

        # 构建第二个文件夹中的文件路径（例如 hmi.ssc.gray.continuum.20111001_120034.png）
        original_filename = base_filename.rsplit('_rect', 1)[0] + ".png"
        original_img_path = os.path.join(folder2, original_filename)

        # 如果第二个文件夹中存在对应的原始图像
        if os.path.exists(original_img_path):
            original_img = Image.open(original_img_path).convert("L")

            # 确保蒙版图像和原始图像的尺寸一致
            if original_img.size == mask_img.size:
                # 处理蒙版图像，保持白色部分（255）并将其他部分（黑色）置为0
                mask_img = mask_img.point(lambda p: p > 128 and 255 or 0)

                # 将蒙版应用到原始图像上
                result_img = Image.composite(original_img, Image.new("L", original_img.size, 0), mask_img)

                # 保存结果图像到第三个文件夹
                result_img.save(os.path.join(folder3, filename))
                print(f"Saved: {os.path.join(folder3, filename)}")
            else:
                print(f"Size mismatch for {filename}, skipping...")
        else:
            print(f"Original image not found for {filename}, skipping...")
