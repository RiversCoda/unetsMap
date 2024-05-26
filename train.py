import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from datetime import datetime
import os
from tqdm import tqdm
from diceLoss import *
from model import *
from myDataLoad import CustomDataset
import matplotlib.pyplot as plt

def train(model, criterion, optimizer, dataloader, device, epoch_losses):
    model.train()
    running_loss = 0.0
    num_batches = len(dataloader)
    pbar = tqdm(enumerate(dataloader), total=num_batches)
    for batch_idx, (inputs, labels) in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        pbar.set_description(f'Loss: {running_loss/(batch_idx+1):.4f}')
    
    epoch_loss = running_loss / num_batches
    epoch_losses.append(epoch_loss)
    return epoch_losses

def save_model(model, epoch, lossName, use_model_name):
    # 确保 models 目录存在
    os.makedirs('models', exist_ok=True)
    
    # 生成模型文件名
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    model_name = f"{lossName}_{use_model_name}_{epoch}_{current_time}.pt"
    model_path = os.path.join('models', model_name)
    
    # 保存模型
    torch.save(model.state_dict(), model_path)

def save_loss_plot(epoch_losses):
    if not os.path.exists('epochLoss'):
        os.makedirs('epochLoss')
    
    plt.figure()
    plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs')
    plt.legend()
    plt.savefig('epochLoss/loss_plot.png')
    plt.close()

if __name__ == "__main__":
    # 设置参数
    num_classes = 6
    in_channels = 3
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 8
    lr = 0.01
    num_epochs = 100
    save_interval = 2  # 保存模型的间隔epoch数
    loss_func = "FocalLoss" # or "DiceLoss" or "CrossEntropyLoss"
    # modelName = "unet" # or "unet_res_ver" or "UNetResSmallVer"
    modelName = "UNetResSmallVer"

    # 数据预处理和加载
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    dataset = CustomDataset(root_dir='trainPics_copy', transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 初始化模型、损失函数和优化器
    if modelName == "unet":
        model = UNet(in_channels, num_classes).to(device)
    elif modelName == "unet_res_ver":
        model = UNetResVer(in_channels, num_classes).to(device)
    elif modelName == "UNetResSmallVer":
        model = UNetResSmallVer(in_channels, num_classes).to(device)

    if loss_func == "DiceLoss":
        criterion = DiceLoss()
    elif loss_func == "CrossEntropyLoss":
      criterion = nn.CrossEntropyLoss()
    elif loss_func == "FocalLoss":
        criterion = FocalLoss()
    else:
        criterion = FocalLoss()

    # optimizer = optim.Adam(model.parameters(), lr=lr)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    # 记录每个epoch的损失
    epoch_losses = []

    # 训练模型
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        epoch_losses = train(model, criterion, optimizer, dataloader, device, epoch_losses)
        
        # 每n个epoch保存一次模型
        if (epoch + 1) % save_interval == 0:
            save_model(model, epoch + 1, loss_func, modelName)

    # 保存损失图
    save_loss_plot(epoch_losses)
