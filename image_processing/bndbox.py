'''

处理了所有带有标注的xml文件，生成对应的二值蒙版

'''

import os
from PIL import Image, ImageDraw
import xml.etree.ElementTree as ET

def process_xml_file(xml_file_path, output_folder):
    # 解析 XML 文件
    tree = ET.parse(xml_file_path)
    root = tree.getroot()

    # 图像尺寸
    image_width = int(root.find("size/width").text)
    image_height = int(root.find("size/height").text)

    # 创建一张黑色图像
    binary_image = Image.new("L", (image_width, image_height), color=0)
    draw = ImageDraw.Draw(binary_image)

    # 遍历每个 <object> 元素
    for obj_elem in root.findall("object"):
        name = obj_elem.find("name").text
        xmin = int(obj_elem.find("bndbox/xmin").text)
        ymin = int(obj_elem.find("bndbox/ymin").text)
        xmax = int(obj_elem.find("bndbox/xmax").text)
        ymax = int(obj_elem.find("bndbox/ymax").text)

        # 在图像上绘制白色矩形
        draw.rectangle([xmin, ymin, xmax, ymax], fill=255)

    # 保存图像
    image_filename = os.path.splitext(os.path.basename(xml_file_path))[0] + ".png"
    output_path = os.path.join(output_folder, image_filename)
    binary_image.save(output_path)
    print(image_filename)

def process_all_xml_files(xml_folder, output_folder):
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    # 获取文件夹中的所有 XML 文件
    xml_files = [f for f in os.listdir(xml_folder) if f.endswith(".xml")]

    # 遍历每个 XML 文件并处理
    for xml_file in xml_files:
        xml_file_path = os.path.join(xml_folder, xml_file)
        process_xml_file(xml_file_path, output_folder)

# 指定 XML 文件夹和输出文件夹
xml_folder = "D:\SunSpotData\label_data\VOC_all\Annotations"
output_folder = "../dataset/bndbox_all"

os.makedirs(output_folder, exist_ok=True)

# 处理所有 XML 文件
process_all_xml_files(xml_folder, output_folder)
