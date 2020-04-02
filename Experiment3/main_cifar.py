import torch
import torch.utils.data as td
import torchvision as tv
import torch.nn as nn
from network import *
import numpy as np
import matplotlib.pyplot as plt


def main():
    train_set = tv.datasets.CIFAR10(
        './data', transform=transform_test, download=True, train=True)
    train_loader = td.DataLoader(train_set, batch_size=64, shuffle=True)

    test_set = tv.datasets.CIFAR10(
        './data', transform=transform_train, download=True, train=False)
    test_loader = td.DataLoader(test_set, batch_size=64)
    device = 'cuda:0'

    net = tv.models.resnet18().to(device) #MyNetwork(10, 3).to(device)
    EPOCHS = 50

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

    torch.save(net.state_dict(), "cifar_res18.pt")

main()
