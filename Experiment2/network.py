import torch
import torch.utils.data as td
import torchvision as tv
import torch.nn as nn

class Squeeze(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.squeeze()


class MyNetwork(nn.Module):
    def __init__(self, cls_num=10, inp=1):
        super().__init__()
        self.convs = nn.ModuleList()
        for i in range(4):
            output = 2 ** (i+3)
            #print(output)
            self.convs.append(
                nn.Sequential(
                    nn.Sequential(nn.LeakyReLU(), 
                    nn.MaxPool2d(2), 
                    nn.BatchNorm2d(inp)) if i > 0 else nn.Sequential(),
                    nn.Conv2d(inp, output, kernel_size=3, padding=1, bias=False),
                    nn.LeakyReLU(),
                    nn.BatchNorm2d(output),
                    nn.Conv2d(output, output, kernel_size=3, padding=1, bias=False),
                )
            )
            inp = output
        self.convs.append(nn.Sequential(nn.AdaptiveAvgPool2d(
            (1, 1)), Squeeze(), nn.Linear(output, cls_num)))

    def forward(self, x, cnt=10):
        #print(x)
        _i = 0
        for conv in self.convs:
            #print(_i)
            x = conv(x)
            #print(x.size())
            _i += 1
            if _i > cnt:
                return x
        return x


transform_train_flip = tv.transforms.Compose([
    tv.transforms.Resize(128, interpolation=2),
    tv.transforms.RandomCrop(128, padding=4),
    # tv.transforms.RandomRotation(3.14/8),
    tv.transforms.RandomHorizontalFlip(),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize((0.5,), (0.5,)),
    #tv.transforms.RandomErasing(),
])

transform_train = tv.transforms.Compose([
    tv.transforms.Resize(128, interpolation=2),
    tv.transforms.RandomCrop(128, padding=4),
    #tv.transforms.RandomCrop(32, padding=4),
    # tv.transforms.RandomRotation(3.14/8),
    # tv.transforms.RandomHorizontalFlip(),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize((0.5,), (0.5,)),
    #tv.transforms.RandomErasing(),
])

transform_test = tv.transforms.Compose([
    tv.transforms.ToTensor(),
    tv.transforms.Normalize((0.5,), (0.5,))
])
