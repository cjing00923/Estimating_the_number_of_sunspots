from PIL import Image, ImageDraw
import os

# 从TXT文件读取数据
def read_file_data(txt_file):
    file_data = {}
    with open(txt_file, 'r') as f:
        lines = f.readlines()
        current_file = None
        for line in lines:
            line = line.strip()
            if line.endswith('.png'):
                current_file = line
                file_data[current_file] = []
            elif current_file and line.startswith('('):
                coordinates = eval(line)
                file_data[current_file].append(coordinates)
    return file_data

# 图像尺寸
image_size = (4096, 4096)

# 输出文件夹
output_folder = "box_all"
os.makedirs(output_folder, exist_ok=True)

# 输入TXT文件路径
txt_file = "box_all/box_all.txt"

# 读取文件内容
file_data = read_file_data(txt_file)

# 生成图像
for filename, rectangles in file_data.items():
    for i, (top_left, bottom_right) in enumerate(rectangles):
        # 创建全黑单通道图像
        img = Image.new("L", image_size, 0)
        draw = ImageDraw.Draw(img)

        # 绘制白色矩形
        draw.rectangle([top_left, bottom_right], fill=255)

        # 保存图像
        base_name = os.path.splitext(filename)[0]
        output_filename = os.path.join(output_folder, f"{base_name}_rect_{i + 1}.png")
        img.save(output_filename)
        print(f"Saved: {output_filename}")
