import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader
import numpy as np
import os
import random
import torch.nn as nn
def set_seed(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

imagesize=32
batchsize=128
maxnum=1024
embed_len=8
channel=1
set_seed(0)
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.Resize((imagesize, imagesize)),
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.Resize((imagesize, imagesize)),
    transforms.ToTensor(),
    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.MNIST(
    root='F:/workspace/pycharm/SFT3/data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=batchsize, shuffle=True, num_workers=0)

testset = torchvision.datasets.MNIST(
    root='F:/workspace/pycharm/SFT3/data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=batchsize, shuffle=False, num_workers=0)
myfile=open('./Input-P0-0','w')

num_class=10
imglist=[]
targetlist=[]

def label2txt(mylist):
    for i in range(len(mylist)):
        myfile.write(str(mylist[i])+' ')
    myfile.write('\n')
    myfile.flush()

def img2txt(mylist):
    print(len(mylist))
    for i in range(0,len(mylist),channel*imagesize*imagesize):
        for j in range(imagesize*imagesize):
            for k in range(channel):
                myfile.write(str(mylist[i+imagesize**2*k+j])+' ')
                # print(str(mylist[i+4*k+j]))
        myfile.write('\n')
    myfile.flush()

def imge2embed(images,pos_embed):
    embed=[]
    for i in range(batchsize):
        loc=0
        for j in range(0,imagesize,embed_len):
            for k in range(0,imagesize,embed_len):
                for l in range(j,j+embed_len):
                    for n in range(k,k+embed_len):
                        embed.append(images[i][0][l][n]+pos_embed[0][loc][(l-j)*embed_len+n-k])
                loc+=1
    return embed

epoch=1
total0 = 0
for i in range(epoch):
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        absolute_pos_embed = nn.Parameter(torch.zeros(1, 16, 64))
        nn.init.trunc_normal_(absolute_pos_embed, std=.02)
        inputs = imge2embed(inputs.tolist(),absolute_pos_embed.tolist())
        imglist += inputs
        targetslist=targets.tolist()
        for tmplab in range(targets.size(0)):
            tmplist = [0] * num_class
            tmplist[targets.tolist()[tmplab]] = 1
            targetlist += tmplist
        total0 += targets.size(0)
        if(total0>=maxnum):
            break

print(total0)
label2txt(targetlist)
img2txt(imglist)

imglist=[]
targetlist=[]
total1 = 0
for i in range(epoch):
    for batch_idx, (inputs, targets) in enumerate(testloader):
        absolute_pos_embed = nn.Parameter(torch.zeros(1, 16, 64))
        nn.init.trunc_normal_(absolute_pos_embed, std=.02)
        inputs = imge2embed(inputs.tolist(),absolute_pos_embed.tolist())
        imglist += inputs
        targetslist = targets.tolist()
        for tmplab in range(targets.size(0)):
            tmplist = [0] * num_class
            tmplist[targets.tolist()[tmplab]] = 1
            targetlist += tmplist
        total1 += targets.size(0)
        if (total1 >= maxnum):
            break

print(total1)
label2txt(targetlist)

img2txt(imglist)

myfile.flush()
myfile.close()