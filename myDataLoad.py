import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import numpy as np
from PIL import Image
import os
from tqdm import tqdm

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_dir = os.path.join(root_dir, 'pics')
        self.label_dir = os.path.join(root_dir, 'label')
        self.images = os.listdir(self.image_dir)
        self.labels = os.listdir(self.label_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.images[idx])
        label_name = os.path.join(self.label_dir, self.labels[idx])

        image = Image.open(img_name)
        label = Image.open(label_name)

        # 将标签调整为与模型输出相同的形状 (6, 512, 512)
        label = label.resize((512, 512), Image.NEAREST)

        # 将RGB颜色转换为对应的类别索引
        label = torch.tensor(self.convert_label_to_index(label))

        if self.transform:
            image = self.transform(image)

        return image, label

    def convert_label_to_index(self, label):
        # 定义颜色与类别索引的映射关系
        color_to_index = {
            (255, 255, 255): 0,  # Impervious surfaces
            (0, 0, 255): 1,       # Building
            (0, 255, 255): 2,     # Low vegetation
            (0, 255, 0): 3,       # Tree
            (255, 255, 0): 4,     # Car
            (255, 0, 0): 5        # Clutter/background
        }

        # 将标签中的RGB颜色转换为类别索引
        # label_index = torch.zeros(label.size(), dtype=torch.long)
        label_height = 512
        label_width  = 512
        label_index = torch.zeros(label_height, label_width, dtype=torch.long)
        for i in range(label_height):
            for j in range(label_height):
                pixel_color = tuple(label.getpixel((i, j)))
                if pixel_color in color_to_index:
                    label_index[i, j] = color_to_index[pixel_color]

        return label_index
transform = transforms.Compose([
    transforms.ToTensor(),
])

# 测试使用CustomDataset读取一张label图片和pics中的一张图片，读取后将他们以数值的格式分别保存到testDate文件夹中，不同维度用,加上换行间隔（需在代码中创建testData文件夹）

if __name__ == "__main__":
    # 创建testData文件夹
    if not os.path.exists('testData'):
        os.makedirs('testData')

    # 设置数据集路径
    root_dir = 'trainPics'
    dataset = CustomDataset(root_dir=root_dir, transform=transform)

    # 读取一张图片和其对应的标签
    image, label = dataset[0]

    # 将图片和标签保存到testData文件夹中
    image_path = 'testData/test_image.csv'
    label_path = 'testData/test_label.csv'

    # 将image转换为numpy数组并保存为csv文件
    image_np = image.numpy().squeeze()
    np.savetxt(image_path, image_np.reshape(-1, image_np.shape[-1]), delimiter=',', fmt='%f')

    # 将label转换为numpy数组并保存为csv文件
    label_np = label.numpy()
    np.savetxt(label_path, label_np, delimiter=',', fmt='%d')