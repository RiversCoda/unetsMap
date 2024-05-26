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
import torch.nn.functional as F

# class DiceLoss(nn.Module):
#     def __init__(self):
#         super(DiceLoss, self).__init__()

#     def forward(self, input, target):
#         smooth = 1e-6
#         input_flat = input.view(-1)
#         target_flat = target.view(-1)
#         intersection = (input_flat * target_flat).sum()
#         dice_coef = (2. * intersection + smooth) / (input_flat.sum() + target_flat.sum() + smooth)
#         return 1 - dice_coef

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, outputs, targets):
        # 将outputs从形状 [N, C, H, W] 转换为 [N, H, W, C]
        outputs = outputs.permute(0, 2, 3, 1).contiguous()
        # 将outputs reshape成 [N * H * W, C]
        outputs = outputs.view(-1, outputs.shape[-1])
        # 将targets reshape成 [N * H * W]
        targets = targets.view(-1)

        # 使用softmax将outputs转换为概率分布
        outputs = F.softmax(outputs, dim=1)

        # 创建one-hot编码的targets
        targets_one_hot = torch.zeros_like(outputs).scatter_(1, targets.unsqueeze(1), 1)

        # 计算Dice系数
        intersection = (outputs * targets_one_hot).sum(dim=0)
        union = outputs.sum(dim=0) + targets_one_hot.sum(dim=0)

        dice = (2. * intersection + self.smooth) / (union + self.smooth)

        return 1 - dice.mean()
    
# class FocalLoss(nn.Module):
#     def __init__(self, alpha=1, gamma=2, reduction='mean'):
#         super(FocalLoss, self).__init__()
#         self.alpha = alpha
#         self.gamma = gamma
#         self.reduction = reduction

#     def forward(self, inputs, targets):
#         # inputs: [batch_size, class_num, pic_height, pic_width]
#         # targets: [batch_size, pic_height, pic_width]
        
#         # Convert targets to one-hot encoding
#         targets = targets.view(-1, 1)
#         one_hot_targets = torch.zeros(targets.size(0), inputs.size(1)).to(inputs.device)
#         one_hot_targets.scatter_(1, targets, 1)
#         one_hot_targets = one_hot_targets.view(inputs.size())
        
#         # Compute softmax over the inputs
#         inputs_soft = F.softmax(inputs, dim=1)
        
#         # Compute the focal loss
#         BCE_loss = F.binary_cross_entropy_with_logits(inputs, one_hot_targets, reduction='none')
#         pt = torch.exp(-BCE_loss)
#         F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        
#         # Apply reduction
#         if self.reduction == 'mean':
#             return F_loss.mean()
#         elif self.reduction == 'sum':
#             return F_loss.sum()
#         else:
#             return F_loss


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, ignore_index=-1):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        # inputs shape: [batch_size, class_num, pic_height, pic_width]
        # targets shape: [batch_size, pic_height, pic_width]
        
        # Convert targets to one-hot encoding
        targets_one_hot = F.one_hot(targets, num_classes=inputs.size(1))  # [batch_size, pic_height, pic_width, class_num]
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2)  # [batch_size, class_num, pic_height, pic_width]

        # Apply softmax to get probabilities
        probs = F.softmax(inputs, dim=1)  # [batch_size, class_num, pic_height, pic_width]
        
        # Focal loss calculation
        pt = torch.where(targets_one_hot == 1, probs, 1 - probs)
        focal_weight = (1 - pt) ** self.gamma

        loss = -self.alpha * focal_weight * torch.log(pt + 1e-8)
        
        # Apply mask for ignored index
        if self.ignore_index is not None:
            mask = targets != self.ignore_index
            loss = loss * mask.unsqueeze(1)

        return loss.mean()