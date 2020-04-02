import torch
import torch.nn.functional as F
import torch.utils.data as td
import torchvision as tv
import torch.nn as nn
from network import *
import matplotlib.pyplot as plt
import numpy as np
from torchsummary import summary
import resnet



class_names = [i for i in range(10)]
def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.5])
    std = np.array([0.5])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    if title is not None:
        plt.title(title)
    plt.imshow(inp)


def optimize(net, step=0, fsize=0, sizes=[32], device='cuda:0', dim_feat=1):
    EPOCHS = 200

    x = nn.Parameter(torch.Tensor(
        1, dim_feat, sizes[0], sizes[0]).to(device).normal_(0, 1e-4))

    for i in range(len(sizes)):
        x = nn.Parameter(F.interpolate(
            x, [sizes[i], sizes[i]], mode='bilinear', align_corners=True)).cuda()
        # plt.imshow((torch.tanh(x).squeeze().detach().cpu().numpy() + 1) / 2)
        # plt.pause(1)
        # optimizer = torch.optim.SGD(
        #     [x], lr=0.01, momentum=0.9, weight_decay=0)
        optimizer = torch.optim.Adam(
            [x], lr=0.01)

        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     optimizer, EPOCHS)
        net.eval()
        # print(output.shape)
        for i in range(EPOCHS):
            optimizer.zero_grad()
            img = F.interpolate(
                x, [sizes[-1], sizes[-1]], mode='bilinear', align_corners=True)

            if step >= 0:
                output = net(torch.tanh(x), step)
                # print(output.shape)
            else:
                output = net(torch.tanh(x))
                # print(output)
            if len(output.shape) == 4:
                N, C, H, W = output.shape
                # + x.abs().mean() * 1e-4# + output[0,:,H//2,W//2].abs().mean() * 1e-3
                # - output[0, fsize, :, :].mean() #
                target =- output[0, fsize, :, :].mean() #- output[0, fsize, H//2, W//2]
            else:
                # + x.abs().mean() * 1e-4# + output[0,:,H//2,W//2].abs().mean() * 1e-3
                target = - F.log_softmax(output.squeeze(), 0)[fsize]
                # print(target)

            target.backward()
            optimizer.step()
            # scheduler.step()
        print(target)

    return (torch.tanh(x).squeeze().detach().cpu().numpy() + 1) / 2

    # plt.imshow((torch.tanh(x).squeeze().detach().cpu().numpy() + 1) / 2)
    # plt.show()


def main():
    # train_set = tv.datasets.MNIST(
    #     './data', transform=transform_train, download=True, train=True)
    # train_loader = td.DataLoader(train_set, batch_size=64, shuffle=True)
    device = 'cuda:0'

    # 对cifar数据
    # net = tv.models.resnet18().to(device)
    # net.load_state_dict(torch.load('cifar_augs_res18.pt'))

    # inp参数对灰度图为1 彩图为3 optimize的dim_feat参数同
    net = MyNetwork(10,1).to(device)
    
    net.load_state_dict(torch.load('mnist_augs_cnn.pt'))

    # W = 2
    # H = 5
    # fig, axs = plt.subplots(W, H, sharex='col', sharey='row',
    #                         gridspec_kw={'wspace': 0,'hspace': 0})
    # for i in range(10):
    #     ax = axs[i % W, i//W]
    #     img = optimize(net, step=-1, fsize=i, sizes=[8,12,16,24,32], device=device, dim_feat=1)
    #     # s = ax.imshow(img.transpose(1, 2, 0))   # 对彩图转置矩阵
    #     s = ax.imshow(img)   
    #     ax.set_xticks([])
    #     ax.set_yticks([])
    #     ax.set_xlabel(i)

    fig, axs = plt.subplots(1, 8, gridspec_kw={'wspace': 0,'hspace': 0})
    for i in range(8):
        ax = axs[i]
        img = optimize(net, step=4, fsize=i, sizes=[8,12,16,24,32], device=device, dim_feat=1)    # [8,12,16,24,32]
        s = ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('filter'+str(i))
    fig.colorbar
    plt.ioff()
    plt.tight_layout()
    plt.show()


main()

# net = tv.models.resnet18(True).cuda()
# img = optimize(net, step=-1, fsize=488,
#                sizes=[8,16,32, 64, 128, 160, 192, 224], device='cuda', dim_feat=3)
# plt.imshow(img.transpose(1, 2, 0))
# # plt.set_xticks([])
# # plt.set_yticks([])
# plt.show()
