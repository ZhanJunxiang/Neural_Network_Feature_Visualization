import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
#import torchvision.datasets.CIFAR10 as Cifar10Dataset
import torchvision

class ResidualBlock(nn.Module):
    """
    子 module: Residual Block ---- ResNet 中一个跨层直连的单元
    """
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        # 如果输入和输出的通道不一致，或其步长不为 1，需要将二者转成一致
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)  # 输出 + 输入
        out = F.relu(out)
        return out


class MYResNet(nn.Module):
    """
    实现主 module: ResNet-18
    ResNet 包含多个 layer, 每个 layer 又包含多个 residual block (上面实现的类)
    因此, 用 ResidualBlock 实现 Residual 部分，用 _make_layer 函数实现 layer
    """
    def __init__(self, ResidualBlock, num_classes=10):
        super(MYResNet, self).__init__()
        self.inchannel = 64
        # 最开始的操作
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        # 四个 layer， 对应 2， 3， 4， 5 层， 每层有两个 residual block
        self.layer1 = self._make_layer(ResidualBlock, 64,  2, stride=1)
        self.layer2 = self._make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(ResidualBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(ResidualBlock, 512, 2, stride=2)
        # 最后的全连接，分类时使用
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, channels, num_blocks, stride):
        """
        构建 layer, 每一个 layer 由多个 residual block 组成
        在 ResNet 中，每一个 layer 中只有两个 residual block
        """
        layers = []
        for i in range(num_blocks):
            if i == 0:  # 第一个是输入的 stride
                layers.append(block(self.inchannel, channels, stride))
            else:    # 后面的所有 stride，都置为 1
                layers.append(block(channels, channels, 1))
            self.inchannel = channels
        return nn.Sequential(*layers)  # 时序容器。Modules 会以他们传入的顺序被添加到容器中。

    def forward(self, x):
        # 最开始的处理
        out = self.conv1(x)
        # 四层 layer
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # 全连接 输出分类信息
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

if __name__=='__main__':
    input = torch.randn((64,3,32,32))
    net = MYResNet(ResidualBlock)
    out = net(input)
    print(out.shape)
    torch.save(net.state_dict(), "/data/mry/code/resnet18-test.pt")
    pass
