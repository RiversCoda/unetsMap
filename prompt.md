1. 数据集格式：
图片路径trainPics/pics
存储的图片命名形如trainPics/pics/top_potsdam_2_10_RGB.tif
标签路径trainPics/label
存储的图片命名形如trainPics/label/top_potsdam_2_10_label.tif
都是512*512的RGB的tif格式图片
图片就是遥感影像，标签是用不同颜色代表对应遥感图像中对应像素所属的类别
其中，标签与类别对应关系是
```
Impervious surfaces (RGB: 255, 255, 255)
Building (RGB: 0, 0, 255)
Low vegetation (RGB: 0, 255, 255)
Tree (RGB: 0, 255, 0)
Car (RGB: 255, 255, 0)
Clutter/background (RGB: 255, 0, 0)
```
2. 我希望分割成6类，如1. 中标签介绍提到的那样
3. 使用Pytorch框架
4. 我需要你帮我实现一个5层的Unet网络，其中修改跳跃连接为add而不是concatenate
5. 我使用本地的4060ti 8G显存训练
6. 我希望你帮我完成你认为有必要的辅助函数
---
以下是额外要求
7. 我希望你帮我实现一个Dice Loss，对比使用交叉熵损失和Dice Loss的效果（在代码中先注释掉Dice Loss的部分，我先使用交叉熵损失实验）
8. 我希望你帮我实现训练进度条