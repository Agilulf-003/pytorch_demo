# image_classification/models/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import NUM_CLASSES

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        """
        一个简单的CNN模型
        """
        super(SimpleCNN, self).__init__()
        # 卷积层 1
        # 输入: 3x128x128, 输出: 16x64x64
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 卷积层 2
        # 输入: 16x64x64, 输出: 32x32x32
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 全连接层
        # 输入: 32 * 32 * 32, 输出: num_classes
        self.fc1 = nn.Linear(32 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        """
        前向传播
        """
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))

        # 展平操作
        x = x.view(-1, 32 * 32 * 32)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

if __name__ == '__main__':
    # 测试一下模型是否能正常工作
    model = SimpleCNN()
    # 创建一个假的输入张量 (batch_size=4, channels=3, height=128, width=128)
    dummy_input = torch.randn(4, 3, 128, 128)
    output = model(dummy_input)
    print(f"模型输出尺寸: {output.size()}") # 应该输出 torch.Size([4, 2])
