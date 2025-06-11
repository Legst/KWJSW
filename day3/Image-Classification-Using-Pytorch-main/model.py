# model.py
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)  # 输入通道3，输出通道16，3x3卷积
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)  # CIFAR10 共10类

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # [B, 16, 16, 16]
        x = self.pool(F.relu(self.conv2(x)))  # [B, 32, 8, 8]
        x = x.view(-1, 32 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
