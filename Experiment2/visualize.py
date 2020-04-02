import torch
import torch.nn.functional as F
import torch.utils.data as td
import torchvision as tv
import torch.nn as nn
from network import *
from resnet18 import *
from vgg11 import *
import matplotlib.pyplot as plt
import numpy as np
from fastai.conv_learner import *
import cv2


import PIL
from cv2 import resize
#%matplotlib inline

class SaveFeatures():
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output):
        self.features = torch.tensor(output,requires_grad=True).cuda()
    def close(self):
        self.hook.remove()


class FilterVisualizer():
    def __init__(self, size=56, upscaling_steps=12, upscaling_factor=1.2):
        self.size, self.upscaling_steps, self.upscaling_factor = size, upscaling_steps, upscaling_factor
        self.model = vgg16(pre=True).cuda().eval()

        #net = MyNetwork(10, 3).cuda()
        #net.load_state_dict(torch.load('cifar_cnn.pt'))

        #net = ResNet(ResidualBlock)
        #net = MYResNet(ResidualBlock)

        net = VGG()
        net.load_state_dict(torch.load('vggtest.pt'))
        self.model = net.cuda().eval()

        set_trainable(self.model, False)

    def visualize(self, layer, filter, lr=0.1, opt_steps=20, blur=None):
        sz = self.size
        img = np.uint8(np.random.uniform(150, 180, (sz, sz, 3))) / 255  # generate random image
        activations = SaveFeatures(list(self.model.children())[0][layer])  # register hook

        for _ in range(self.upscaling_steps):  # scale the image up upscaling_steps times
            train_tfms, val_tfms = tfms_from_model(vgg16, sz)
            img_var = V(val_tfms(img)[None], requires_grad=True)  # convert image to Variable that requires grad
            optimizer = torch.optim.Adam([img_var], lr=lr, weight_decay=1e-6)
            for n in range(opt_steps):  # optimize pixel values for opt_steps times
                optimizer.zero_grad()
                self.model(img_var)
                loss = -activations.features[0, filter].mean()
                loss.backward()
                optimizer.step()
            img = val_tfms.denorm(img_var.data.cpu().numpy()[0].transpose(1, 2, 0))
            self.output = img
            sz = int(self.upscaling_factor * sz)  # calculate new image size
            img = cv2.resize(img, (sz, sz), interpolation=cv2.INTER_CUBIC)  # scale image up
            if blur is not None: img = cv2.blur(img, (blur, blur))  # blur image to reduce high frequency patterns
        self.save(layer, filter)
        '''
        plt.figure(figsize=(7, 7))
        plt.title("layer_" + str(layer) + "_filter_" + str(filter) + ".jpg")
        plt.imshow(np.clip(self.output, 0, 1))
        plt.show()
        '''
        fft(np.clip(self.output, 0, 1), layer, filter)
        activations.close()

    def save(self, layer, filter):
        plt.imsave("layer_" + str(layer) + "_filter_" + str(filter) + ".jpg", np.clip(self.output, 0, 1))

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


def fft(cv2_img, layer, filter):
    import cv2 as cv
    import numpy as np
    from matplotlib import pyplot as plt

    def rgb2gray(rgb):
        return np.dot(rgb[..., :3], [0.114, 0.587, 0.299])

    img = rgb2gray(cv2_img)
    # 读取图像
    #img = cv.imread('test.png', 0)

    # 快速傅里叶变换算法得到频率分布
    f = np.fft.fft2(img)

    # 默认结果中心点位置是在左上角,
    # 调用fftshift()函数转移到中间位置
    fshift = np.fft.fftshift(f)

    # fft结果是复数, 其绝对值结果是振幅
    fimg = np.log(np.abs(fshift))

    # 展示结果
    plt.figure()
    plt.subplot(121), plt.imshow(img, 'gray'), plt.title('Original Fourier')
    plt.axis('off')
    plt.subplot(122), plt.imshow(fimg, 'gray'), plt.title('Fourier Fourier')
    plt.axis('off')
    plt.show()
    plt.savefig("Fourier-layer_" + str(layer) + "_filter_" + str(filter) + ".jpg")

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
                output = net(torch.tanh(img), step)
            else:
                output = net(torch.tanh(x))
            if len(output.shape) == 4:
                N, C, H, W = output.shape
                # + x.abs().mean() * 1e-4# + output[0,:,H//2,W//2].abs().mean() * 1e-3
                # - output[0, fsize, :, :].mean() #
                target =- output[0, fsize, :, :].mean() #- output[0, fsize, H//2, W//2]
            else:
                # + x.abs().mean() * 1e-4# + output[0,:,H//2,W//2].abs().mean() * 1e-3
                target = - F.log_softmax(output.squeeze(), 0)[fsize]

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

    net = MyNetwork(10,3).to(device)

    net.load_state_dict(torch.load('cifar_cnn.pt'))

    W = 2
    H = 5
    fig, axs = plt.subplots(W, H, sharex='col', sharey='row',
                            gridspec_kw={'wspace': 0,'hspace': 0})
    for i in range(10):
        ax = axs[i % W, i//W]
        img = optimize(net, step=4, fsize=i, sizes=[8,12,16,24,32], device=device, dim_feat=3)
        s = ax.imshow(img.transpose(1, 2, 0))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(i)
    fig.colorbar
    plt.ioff()
    plt.tight_layout()
    plt.show()


#main()

# net = tv.models.resnet18(True).cuda()
# img = optimize(net, step=-1, fsize=488,
#                sizes=[8,16,32, 64, 128, 160, 192, 224], device='cuda', dim_feat=3)
# plt.imshow(img.transpose(1, 2, 0))
# # plt.set_xticks([])
# # plt.set_yticks([])
# plt.show()

if __name__=='__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Vis of filter-th of layer-th ")
    parser.add_argument("--layer", type=int)
    parser.add_argument("--filter", type=int)
    args = parser.parse_args()

    layer = args.layer
    filter = args.filter

    FV = FilterVisualizer(size=128, upscaling_steps=2, upscaling_factor=1.2)
    FV.visualize(layer, filter, blur=5)

    img = PIL.Image.open("layer_" + str(layer) + "_filter_" + str(filter) + ".jpg")
    plt.figure(figsize=(7,7))
    plt.title("layer_" + str(layer) + "_filter_" + str(filter) + ".jpg")
    plt.imshow(img)
    plt.show()
