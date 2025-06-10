import torch
import torchvision.transforms as T
from PIL import Image
from model import *  # 自定义模型模块

# 读取图片路径
img_path = "Image/real.png"
img = Image.open(img_path)

# PNG 图片默认有四个通道（RGBA），转为RGB三通道
img = img.convert("RGB")

# 定义预处理流程：调整大小并转为Tensor
preprocess = T.Compose([
    T.Resize((32, 32)),
    T.ToTensor()
])
img_tensor = preprocess(img)
print(f"图片张量尺寸: {img_tensor.shape}")

# 加载预训练模型，映射到CPU上
net = torch.load("model_save/chen_3.pth", map_location="cpu")

# 调整输入维度，增加batch维度
img_tensor = img_tensor.unsqueeze(0)  # shape: (1, 3, 32, 32)

# 切换模型到评估模式，禁用梯度计算
net.eval()
with torch.no_grad():
    pred = net(img_tensor)

print(f"预测类别索引: {pred.argmax(dim=1).item()}")
