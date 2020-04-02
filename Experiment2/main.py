import torch
import torch.utils.data as td
import torchvision as tv
import torch.nn as nn
from network import *
import numpy as np
import matplotlib.pyplot as plt


def main():
    train_set = tv.datasets.MNIST(
        './data', transform=transform_train, download=True, train=True)
    train_loader = td.DataLoader(train_set, batch_size=128, shuffle=True)

    test_set = tv.datasets.MNIST(
        './data', transform=transform_train, download=True, train=False)
    test_loader = td.DataLoader(test_set, batch_size=128)
    device = 'cuda:0'

    net = MyNetwork().to(device)
    EPOCHS = 20

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, EPOCHS, eta_min=1e-6, last_epoch=-1)
    

    # Training
    def train(epoch):
        print('\nEpoch: %d' % epoch)
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        print('[TRAIN]Loss: %.3f | Acc: %.3f%% (%d/%d)'
              % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    def test(epoch):
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        print('[TEST] Loss: %.3f | Acc: %.3f%% (%d/%d)'
              % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    for i in range(EPOCHS):
        train(i)
        test(i)
        scheduler.step()

    torch.save(net.state_dict(), "mnist_cnn.pt")

main()

# class_names = [i for i in range(10)]
# def imshow(inp, title=None):
#     """Imshow for Tensor."""
#     inp = inp.numpy().transpose((1, 2, 0))
#     mean = np.array([0.5])
#     std = np.array([0.5])
#     inp = std * inp + mean
#     inp = np.clip(inp, 0, 1)
#     if title is not None:
#         plt.title(title)
#     plt.imshow(inp)
    
#     #plt.pause(10)  # pause a bit so that plots are updated


# # Get a batch of training data
# train_set = tv.datasets.MNIST(
#         './data', transform=transform_train, download=True, train=True)
# train_loader = td.DataLoader(train_set, batch_size=64, shuffle=True)
# inputs, classes = next(iter(train_loader))

# # Make a grid from batch
# out = tv.utils.make_grid(inputs)

# imshow(out, title=[class_names[x] for x in classes])
# plt.show()
