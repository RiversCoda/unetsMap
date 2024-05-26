import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
from model import UNet
from myDataLoad import *

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
            for i in range(outputs.shape[1]):
                output_path_i = os.path.join(output_dir, f"outputs_{idx}_dim{i}.txt")
                np.savetxt(output_path_i, outputs[:, i, :, :].cpu().numpy().flatten(), fmt='%f')

            # _, predicted = torch.max(outputs, 1)  # 获取预测结果中概率最大的类别
            predicted = torch.argmax(outputs, dim=1)
            print(f"Predicted shape: {predicted.shape}, unique values: {predicted.unique()}")

            output_path2 = os.path.join(output_dir, f"predicts_{idx}.txt")
            np.savetxt(output_path2, predicted.cpu().numpy().flatten(), fmt='%f')

            total_correct += (predicted == labels).sum().item()
            total_pixels += labels.numel()

            # 将预测结果映射回标签格式
            predicted_labels = torch.zeros_like(labels)
            for i in range(predicted_labels.shape[0]):
                for j in range(predicted_labels.shape[1]):
                    predicted_labels[i, j] = outputs[i, :, j, j].argmax().item()

            # 存储预测结果
            for i in range(predicted_labels.shape[0]):
                prediction = predicted_labels[i].cpu().numpy()
                prediction_img = Image.fromarray(prediction.astype(np.uint8))
                prediction_img.save(os.path.join(output_dir, f"prediction_{i}.tif"))

    accuracy = total_correct / total_pixels
    return accuracy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet(3, 6)

# 使用示例
test_dataset = CustomDataset(root_dir='testPics', transform=transform)
test_dataloader = DataLoader(test_dataset, batch_size=1)
output_directory = 'test_output'
os.makedirs(output_directory, exist_ok=True)

model.load_state_dict(torch.load('DiceLoss_unet_20_2024-05-25_15-56-56.pt'))
model.to(device)

accuracy = test(model, test_dataloader, device, output_directory)
print(f"Accuracy: {accuracy}")
