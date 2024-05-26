import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
from model import *
from myDataLoad import *

# 定义类别标签到颜色的映射
label_to_color = {
    0: (255, 255, 255),  # Impervious surfaces
    1: (0, 0, 255),      # Building
    2: (0, 255, 255),    # Low vegetation
    3: (0, 255, 0),      # Tree
    4: (255, 255, 0),    # Car
    5: (255, 0, 0)       # Clutter/background
}

def test(model, dataloader, device, output_dir):
    model.eval()
    total_correct = 0
    total_pixels = 0
    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            # 输出outputs形状
            print(f"Outputs shape: {outputs.shape}")
            print(f"Outputs max: {outputs.max()}, min: {outputs.min()}, mean: {outputs.mean()}")
            
            # 保存outputs为txt格式
            output_path = os.path.join(output_dir, f"outputs_{idx}.txt")
            
            # 分别保存outputs的dim1的6个通道
            # for i in range(outputs.shape[1]):
            #     output_path_i = os.path.join(output_dir, f"outputs_{idx}_dim{i}.txt")
            #     np.savetxt(output_path_i, outputs[:, i, :, :].cpu().numpy().flatten(), fmt='%f')

            # 获取预测结果
            predicted = torch.argmax(outputs, dim=1)
            print(f"Predicted shape: {predicted.shape}, unique values: {predicted.unique()}")

            # output_path2 = os.path.join(output_dir, f"predicts_{idx}.txt")
            # np.savetxt(output_path2, predicted.cpu().numpy().flatten(), fmt='%f')

            total_correct += (predicted == labels).sum().item()
            total_pixels += labels.numel()

            # 将预测结果映射到颜色
            for i in range(predicted.shape[0]):
                prediction = predicted[i].cpu().numpy()
                prediction_img = np.zeros((prediction.shape[0], prediction.shape[1], 3), dtype=np.uint8)
                for label, color in label_to_color.items():
                    prediction_img[prediction == label] = color
                prediction_img = Image.fromarray(prediction_img)
                prediction_img.save(os.path.join(output_dir, f"prediction_{idx}_{i}.tif"))

    accuracy = total_correct / total_pixels
    return accuracy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = UNetResVer(3, 6).to(device)
model = UNetResSmallVer(3, 6).to(device)
# model = UNet(3, 6).to(device)

# 使用示例
test_dataset = CustomDataset(root_dir='testPics', transform=transform)
test_dataloader = DataLoader(test_dataset, batch_size=1)
output_directory = 'test_output'
os.makedirs(output_directory, exist_ok=True)

model.load_state_dict(torch.load('models\FocalLoss_UNetResSmallVer_46_2024-05-26_13-52-19.pt'))
model.to(device)

accuracy = test(model, test_dataloader, device, output_directory)
print(f"Accuracy: {accuracy}")
