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

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
    
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = DoubleConv(64, 128)
        self.down2 = DoubleConv(128, 256)
        self.down3 = DoubleConv(256, 512)
        self.down4 = DoubleConv(512, 1024)
        self.pool = nn.MaxPool2d(2)  # 添加池化操作
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.r1 = DoubleConv(1024, 512)
        self.r2 = DoubleConv(512, 256)
        self.r3 = DoubleConv(256, 128)
        self.r4 = DoubleConv(128, 64)
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        x1c = self.inc(x)
        # print("x1c shape: ", x1c.shape)

        x2 = self.pool(x1c)
        # print("x2 shape: ", x2.shape)
        x2c = self.down1(x2)
        # print("x2c shape: ", x2c.shape)

        x3 = self.pool(x2c)
        # print("x3 shape: ", x3.shape)
        x3c = self.down2(x3)
        # print("x3c shape: ", x3c.shape)

        x4 = self.pool(x3c)
        x4c = self.down3(x4)

        x5 = self.pool(x4c)
        x5c = self.down4(x5)

        x4u = self.up1(x5c)
        x4ucat = torch.cat([x4c, x4u], dim=1)
        x4ucatc = self.r1(x4ucat)

        x3u = self.up2(x4ucatc)
        x3ucat = torch.cat([x3c, x3u], dim=1)
        x3ucatc = self.r2(x3ucat)

        x2u = self.up3(x3ucatc)
        x2ucat = torch.cat([x2c, x2u], dim=1)
        x2ucatc = self.r3(x2ucat)

        x1u = self.up4(x2ucatc)
        x1ucat = torch.cat([x1c, x1u], dim=1)
        x1ucatc = self.r4(x1ucat)
        # print("x1ucatc shape: ", x1ucatc.shape)
        # [1, 64, 512, 512]
        output = self.out_conv(x1ucatc)
        return output
        

class UNetResVer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetResVer, self).__init__()
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = DoubleConv(64, 128)
        self.down2 = DoubleConv(128, 256)
        self.down3 = DoubleConv(256, 512)
        self.down4 = DoubleConv(512, 1024)
        self.pool = nn.MaxPool2d(2)  # 添加池化操作
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        # self.r1 = DoubleConv(1024, 512)
        self.r2 = DoubleConv(512, 512)
        self.r3 = DoubleConv(256, 256)
        self.r4 = DoubleConv(128, 128)
        self.r5 = DoubleConv(64, 64)
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        x1c = self.inc(x)
        # print("x1c shape: ", x1c.shape)

        x2 = self.pool(x1c)
        x2c = self.down1(x2)

        x3 = self.pool(x2c)
        x3c = self.down2(x3)

        x4 = self.pool(x3c)
        x4c = self.down3(x4)

        x5 = self.pool(x4c)
        x5c = self.down4(x5)

        x4u = self.up1(x5c)
        # x4ucat = torch.cat([x4c, x4u], dim=1)\
        # x4ucatc = self.r1(x4ucat)
        x4uadd = torch.add(x4c, x4u)
        x4uaddc = self.r2(x4uadd)
        

        # x3u = self.up2(x4ucatc)
        # x3ucat = torch.cat([x3c, x3u], dim=1)
        # x3ucatc = self.r2(x3ucat)
        x3u = self.up2(x4uaddc)
        x3uadd = torch.add(x3c, x3u)
        x3uaddc = self.r3(x3uadd)

        # x2u = self.up3(x3ucatc)
        # x2ucat = torch.cat([x2c, x2u], dim=1)
        # x2ucatc = self.r3(x2ucat)
        x2u = self.up3(x3uaddc)
        x2uadd = torch.add(x2c, x2u)
        x2uaddc = self.r4(x2uadd)

        # x1u = self.up4(x2ucatc)
        # x1ucat = torch.cat([x1c, x1u], dim=1)
        # x1ucatc = self.r4(x1ucat)
        x1u = self.up4(x2uaddc)
        x1uadd = torch.add(x1c, x1u)
        x1uaddc = self.r5(x1uadd)

        # print("x1uaddc shape: ", x1uaddc.shape)
        # [1, 64, 512, 512]
        # output = self.out_conv(x1ucatc)
        output = self.out_conv(x1uaddc)
        return output



class UNetResSmallVer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetResSmallVer, self).__init__()
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = DoubleConv(64, 128)
        self.down2 = DoubleConv(128, 256)
        self.down3 = DoubleConv(256, 512)
        self.down4 = DoubleConv(512, 1024)
        self.pool = nn.MaxPool2d(2)  # 添加池化操作
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        # self.r1 = DoubleConv(1024, 512)
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        x1c = self.inc(x)
        # print("x1c shape: ", x1c.shape)

        x2 = self.pool(x1c)
        x2c = self.down1(x2)

        x3 = self.pool(x2c)
        x3c = self.down2(x3)

        x4 = self.pool(x3c)
        x4c = self.down3(x4)

        x5 = self.pool(x4c)
        x5c = self.down4(x5)

        x4u = self.up1(x5c)
        x4uadd = torch.add(x4c, x4u)

        x3u = self.up2(x4uadd)
        x3uadd = torch.add(x3c, x3u)

        x2u = self.up3(x3uadd)
        x2uadd = torch.add(x2c, x2u)

        x1u = self.up4(x2uadd)
        x1uadd = torch.add(x1c, x1u)

        output = self.out_conv(x1uadd)
        return output