import torch
import torch.nn.functional as F

# 定义输入张量（5x5）
x = torch.tensor([[1, 2, 0, 3, 1],
                  [0, 1, 2, 3, 1],
                  [1, 2, 1, 0, 0],
                  [5, 2, 3, 1, 1],
                  [2, 1, 0, 1, 1]])

# 定义卷积核（3x3）
filter = torch.tensor([[1, 2, 1],
                       [0, 1, 0],
                       [2, 1, 0]])

# 原始形状输出
print("Input shape:", x.shape)
print("Kernel shape:", filter.shape)

# 调整形状以符合conv2d输入要求: (batch_size, channels, height, width)
x = x.view(1, 1, 5, 5)
filter = filter.view(1, 1, 3, 3)
print("Reshaped input:", x.shape)
print("Reshaped kernel:", filter.shape)

# 无填充，步长为1的卷积
out1 = F.conv2d(x, filter, stride=1)
print("Output stride=1:\n", out1)

# 无填充，步长为2的卷积
out2 = F.conv2d(x, filter, stride=2)
print("Output stride=2:\n", out2)

# 使用padding=1，步长为1，保持输入尺寸
out3 = F.conv2d(x, filter, stride=1, padding=1)
print("Output with padding=1:\n", out3)
