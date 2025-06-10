# CIFAR10数据集上的完整训练流程示例
import time
import torch
import torchvision
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model import *  # 自定义模型文件

# 数据集准备
train_dataset = torchvision.datasets.CIFAR10(root="../dataset_chen",
                                             train=True,
                                             transform=torchvision.transforms.ToTensor(),
                                             download=True)

test_dataset = torchvision.datasets.CIFAR10(root="../dataset_chen",
                                            train=False,
                                            transform=torchvision.transforms.ToTensor(),
                                            download=True)

print(f"训练集样本数量: {len(train_dataset)}")
print(f"测试集样本数量: {len(test_dataset)}")

# 数据加载器
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 初始化模型
net = Chen()

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
lr = 0.01
optimizer = optim.SGD(net.parameters(), lr=lr)

# 训练参数
num_epochs = 10
train_steps = 0
test_steps = 0

# TensorBoard日志
writer = SummaryWriter("../logs_train")

start_time = time.time()

for epoch in range(num_epochs):
    print(f"==== 开始第{epoch + 1}轮训练 ====")
    net.train()
    for batch in train_loader:
        inputs, labels = batch
        outputs = net(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_steps += 1
        if train_steps % 500 == 0:
            print(f"第{train_steps}步训练损失: {loss.item():.4f}")
            writer.add_scalar("Train/Loss", loss.item(), train_steps)

    elapsed = time.time() - start_time
    print(f"本轮训练用时: {elapsed:.2f}秒")

    net.eval()
    total_loss = 0.0
    total_correct = 0
    with torch.no_grad():
        for batch in test_loader:
            inputs, labels = batch
            preds = net(inputs)
            loss = criterion(preds, labels)
            total_loss += loss.item()

            correct = (preds.argmax(dim=1) == labels).sum().item()
            total_correct += correct

    avg_loss = total_loss
    accuracy = total_correct / len(test_dataset)
    print(f"测试集总损失: {avg_loss:.4f}")
    print(f"测试集准确率: {accuracy:.4f}")

    writer.add_scalar("Test/Loss", avg_loss, test_steps)
    writer.add_scalar("Test/Accuracy", accuracy, test_steps)
    test_steps += 1

    torch.save(net, f"model_save/chen_{epoch}.pth")
    print("模型已保存\n")

writer.close()
