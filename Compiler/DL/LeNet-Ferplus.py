# External Libraries
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from PIL import Image
import numpy as np
import torch
import torchvision
from tqdm import tqdm

# Standard Libraries
from os import path, makedirs
import copy

# Modules
from fer_utils import udata

import os
import argparse

# from utils import progress_bar
import random



parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

device = 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch


# Data
print('==> Preparing data..')

base_path_to_dataset='./data/FER_Plus/'

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    # transforms.Resize((32,32)),
    # transforms.RandomCrop(32, padding=0),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
# transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# data_transforms = [transforms.Resize((32,32)),transforms.ColorJitter(brightness=0.5, contrast=0.5),
#                        transforms.RandomHorizontalFlip(p=0.5),
#                        transforms.RandomAffine(degrees=30,
#                                                translate=(.1, .1),
#                                                scale=(1.0, 1.25),
#                                                resample=Image.BILINEAR)]
# data_transforms = [transforms.Resize((32,32)),transforms.ColorJitter(brightness=0.5, contrast=0.5),
#                        transforms.RandomHorizontalFlip(p=0.5)]
data_transforms = [transforms.Resize((32,32)),transforms.RandomHorizontalFlip(p=0.5)]
data_transforms_test = [transforms.Resize((32,32))]

val_data = udata.FERplus(idx_set=1,
                             max_loaded_images_per_label=100000,
                             transforms=transforms.Compose(data_transforms_test),
                             base_path_to_FER_plus=base_path_to_dataset)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

test_data = udata.FERplus(idx_set=2,
                             max_loaded_images_per_label=100000,
                             transforms=transforms.Compose(data_transforms_test),
                             base_path_to_FER_plus=base_path_to_dataset)
testloader = DataLoader(val_data, batch_size=128, shuffle=False)

train_data = udata.FERplus(idx_set=0,
                               max_loaded_images_per_label=500000,
                               transforms=transforms.Compose(data_transforms),
                               base_path_to_FER_plus=base_path_to_dataset)

trainloader = DataLoader(train_data, batch_size=256, shuffle=True)

classes = ('anger','disgust','fear','happy','neutral','sad','surprised')

# Model
print('==> Building model..')
net = nn.Sequential(
    nn.Conv2d(3, 20, 5),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(20, 50, 5),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Flatten(),
    nn.ReLU(),
    nn.Linear(1250, 500),
    nn.ReLU(),
    nn.Linear(500, 8)
)
# from torch.nn import init
# for name, param in net.named_parameters():
#     if 'weight' in name:
#         init.normal_(param, mean=0, std=0.01) # 正态初始化
#     if 'bias' in name:
#         init.constant_(param, val=0)
net = net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,momentum=0.9, weight_decay=5e-4)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 20, gamma=0.1, last_epoch=-1)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
# optimizer=optim.AdamW(net.parameters(),lr=0.0001,weight_decay=5e-8)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # print(inputs.shape)
        # inputs=inputs.expand(-1, 3, -1, -1)
        inputs, targets = inputs.to(device), targets.to(device)
        # print(inputs.shape)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    print('Loss: %.3f | Acc: %.3f%% (%d/%d)'% (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            # inputs = inputs.expand(-1, 3, -1, -1)
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        print('Loss: %.3f | Acc: %.3f%% (%d/%d)'% (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        torch.save(net, './checkpoint/ckpt_ferplus_lenet.pth')
        best_acc = acc

for epoch in tqdm(range(start_epoch, start_epoch+10)):
    # if (epoch % 10 == 0 and epoch != 0):
    #     optimizer = optim.SGD(net.parameters(), lr=args.lr * 0.1, momentum=0.9)
    #     args.lr*=0.1
    train(epoch)
    test(epoch)
    # scheduler.step()