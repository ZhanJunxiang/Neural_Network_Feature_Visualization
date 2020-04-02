import torch.optim as optim
import torch.utils.data
import torch.backends.cudnn as cudnn
from torchvision import transforms as transforms
import torch.nn as nn
import numpy as np
import torchvision

class VGG(nn.Module):

    def __init__(self, num_classes=10):
        super(VGG, self).__init__()
        self.features = nn.Sequential(
            # 第一层
            # 取卷积核为3*3，补边为1，输入通道为3，输出通道为64
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            # 批标准化
            # nn.BatchNorm2d(64, affine=True),
            # 激活函数
            nn.ReLU(inplace=True),
            # 池化，核为2*2，步长为2
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 第二层
            # 取卷积核为3*3，补边为1，输入通道为64，输出通道为128
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            # nn.BatchNorm2d(128, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 第三层
            # 取卷积核为3*3，补边为1，输入通道为128，输出通道为256
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            # nn.BatchNorm2d(256, affine=True),
            nn.ReLU(inplace=True),
            # 取卷积核为3*3，补边为1，输入通道为256，输出通道为256
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            # nn.BatchNorm2d(256, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 第四层
            # 取卷积核为3*3，补边为1，输入通道为256，输出通道为512
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            # nn.BatchNorm2d(512, affine=True),
            nn.ReLU(inplace=True),
            # 取卷积核为3*3，补边为1，输入通道为512，输出通道为512
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            # nn.BatchNorm2d(512, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 第五层
            # 取卷积核为3*3，补边为1，输入通道为512，输出通道为512
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            # nn.BatchNorm2d(512, affine=True),
            nn.ReLU(inplace=True),
            # 取卷积核为3*3，补边为1，输入通道为512，输出通道为512
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            # nn.BatchNorm2d(512, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            # 一层全连接层，输入512层，输出10（10类）
            nn.Linear(8192, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

if __name__=='__main__':
    input = torch.randn((64,3,32,32))
    net = VGG()
    out = net(input)
    print(out.shape)
    torch.save(net.state_dict(), "vggtest.pt")
    pass
