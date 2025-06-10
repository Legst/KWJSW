import torch
import torch.nn as nn

class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 32, 5, padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, 5, padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),  # 32x32经过三次2x2池化后变成4x4
            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.layers(x)

if __name__ == '__main__':
    net = NeuralNet()
    test_input = torch.ones((64, 3, 32, 32))
    result = net(test_input)
    print(result.shape)

