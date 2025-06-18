import os

import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.nn import functional as F
from torchvision.utils import save_image

# 自定义数据集类
class CustomDataset(Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.image_files = sorted(os.listdir(image_folder))
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image_file = self.image_files[index]
        image_path = os.path.join(self.image_folder, image_file)

        try:
            image = Image.open(image_path).convert('L')
        except Exception as e:
            print(f"Error loading image from {image_path}: {e}")
            return None

        # 将图像的像素值缩放到 [0, 1] 范围
        image = np.array(image).astype(np.float32) / 255.0

        if self.transform is not None:
            image = self.transform(image)

        return image


class Conv_Block(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(Conv_Block, self).__init__()
        self.layer=nn.Sequential(
            nn.Conv2d(in_channel,out_channel,3,1,1,padding_mode='reflect',bias=False),
            nn.GroupNorm(32, out_channel, eps=1e-05, affine=True),
            nn.Dropout2d(0.3),
            nn.LeakyReLU(),
            nn.Conv2d(out_channel, out_channel, 3, 1, 1, padding_mode='reflect', bias=False),
            nn.GroupNorm(32, out_channel, eps=1e-05, affine=True),
            nn.Dropout2d(0.3),
            nn.LeakyReLU()
        )
    def forward(self,x):
        return self.layer(x)

class DownSample(nn.Module):
    def __init__(self,channel):
        super(DownSample, self).__init__()
        self.layer=nn.Sequential(
            nn.Conv2d(channel,channel,3,2,1,padding_mode='reflect',bias=False),
            nn.GroupNorm(32, channel, eps=1e-05, affine=True),
            nn.LeakyReLU()
        )
    def forward(self,x):
        return self.layer(x)

class UpSample(nn.Module):
    def __init__(self,channel):
        super(UpSample, self).__init__()
        self.layer=nn.Conv2d(channel,channel//2,1,1)
    def forward(self,x,feature_map):
        up=F.interpolate(x,scale_factor=2,mode='nearest')
        out=self.layer(up)
        return torch.cat((out,feature_map),dim=1)

# 定义UNET神经网络模型
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.c1=Conv_Block(1,64)
        self.d1=DownSample(64)
        self.c2=Conv_Block(64,128)
        self.d2=DownSample(128)
        self.c3=Conv_Block(128,256)
        self.d3=DownSample(256)
        self.c4=Conv_Block(256,512)
        self.d4=DownSample(512)
        self.c5=Conv_Block(512,1024)
        self.u1=UpSample(1024)
        self.c6=Conv_Block(1024,512)
        self.u2=UpSample(512)
        self.c7=Conv_Block(512,256)
        self.u3=UpSample(256)
        self.c8=Conv_Block(256,128)
        self.u4=UpSample(128)
        self.c9=Conv_Block(128,64)
        self.out = nn.Conv2d(64, 1, 3, 1, padding=1)
        self.Th=nn.Sigmoid()

    def forward(self, x):
        R1=self.c1(x)
        R2=self.c2(self.d1(R1))
        R3=self.c3(self.d2(R2))
        R4=self.c4(self.d3(R3))
        R5=self.c5(self.d4(R4))
        O1=self.c6(self.u1(R5,R4))
        O2=self.c7(self.u2(O1,R3))
        O3=self.c8(self.u3(O2,R2))
        O4=self.c9(self.u4(O3,R1))

        return self.Th(self.out(O4))



import time  # 添加时间模块

def predict_images(dataset, model, data_loader, output_folder, device,):
    model.eval()
    model.to(device)

    with torch.no_grad():
        for i, images in enumerate(data_loader):
            start_time = time.time()  # 开始计时

            images = images.to(device)
            H, W = images.size(2), images.size(3)
            full_image = torch.zeros(1, 1, 4096, 4096).to(device)

            stride = 1024
            for h in range(0, H, stride):
                for w in range(0, W, stride):
                    image_patch = images[:, :, h:h + stride, w:w + stride]
                    output_patch = model(image_patch)
                    full_image[:, :, h:h + stride, w:w + stride] += output_patch

            binary_output = (full_image > 0.9).float()
            pil_image = transforms.ToPILImage()(binary_output.squeeze().cpu())
            original_filename = dataset.image_files[i]
            save_path = os.path.join(output_folder, f'{original_filename}')
            pil_image.save(save_path)

            end_time = time.time()  # 结束计时
            elapsed_time = end_time - start_time
            print(f"{original_filename} processed in {elapsed_time:.2f} seconds.")


def predict_run(image_folder,  output_folder, model_path, batch_size=1):
    # 创建预测结果文件夹
    os.makedirs(output_folder, exist_ok=True)

    # 定义数据预处理的转换器
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # 创建自定义数据集实例
    dataset = CustomDataset(image_folder,  transform)

    # 创建数据加载器
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # 创建UNET神经网络模型实例
    model = UNet()

    # 加载训练好的模型参数
    model.load_state_dict(torch.load(model_path))

    # 在测试集上进行预测
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predict_images(dataset, model, data_loader, output_folder, device)



if __name__ == "__main__":
    # 图像文件夹文件夹路径

    image_folder = "dataset/..."
    # 预测结果文件夹路径
    output_folder = "dataset/..."
    # 训练好的模型参数文件路径
    model_path = "runs/..."
    # 运行预测
    predict_run(image_folder, output_folder, model_path, batch_size=1)