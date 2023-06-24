import torch.backends.cudnn as cudnn
import torchvision
import os
from torch.utils.data import Dataset,DataLoader
from CK_Plus_DataSet import *
from fer_utils import udata
from VGG import *
import random
import torch
from torchvision import transforms


device = 'cpu'
print('==> Preparing data..')

transform_train3 = transforms.Compose([
    transforms.Resize((32, 32)),
    # transforms.RandomCrop(32, padding=0),
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test3 = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_train1 = transforms.Compose([
    transforms.Resize((32,32)),
    # transforms.RandomCrop(32, padding=0),
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4822), (0.1994)),
])

transform_test1 = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Normalize((0.4822), (0.1994)),
])

print('==> Building model..')
net=torch.load('./checkpoint/ckpt_vgg16.pth',map_location='cpu')
# net=torch.load('./checkpoint/ckpt_imagenet.pth')
net = net.to(device)

flist=[]
featurelen = 512

transform_train = transforms.Compose([
    transforms.Resize((32, 32)),
    # transforms.RandomCrop(32, padding=0),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4822), (0.1994))
])

transform_test = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.4822), (0.1994))
])
# Data
print('==> Preparing data..')
trainset =CK(split = 'Training', fold = 1, transform=transform_train)
trainloader = DataLoader(trainset,batch_size=256,shuffle=True,drop_last=False)

testset = CK(split = 'Testing', fold = 1, transform=transform_train)
testloader = DataLoader(testset,batch_size=100,shuffle=False,drop_last=False)

classes = ('anger','disgust','fear','happy','neutral','sad','surprised')

n_classes0 = 7
fealist0_0 = [torch.tensor([0.0] * featurelen) for i in range(n_classes0)]
cntlist0 = [0] * n_classes0
wlist0=[0.0]*n_classes0

total=0
O=torch.tensor([0.0] * featurelen)
with torch.no_grad():
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs = inputs.expand(-1, 3, -1, -1)
        inputs, targets = inputs.to(device), targets.squeeze().long().to(device)
        tmpfeature=net.forward_getfeature(inputs)
        tmplabels = targets.tolist()
        labellen = len(tmplabels)
        O+=tmpfeature.sum(dim=0).cpu()
        total += targets.size(0)
        for i in range(labellen):
            # if (cntlist0[tmplabels[i]] == 100):
            #     continue
            cntlist0[tmplabels[i]] += 1
            fealist0_0[tmplabels[i]] += tmpfeature[i].cpu()
O=O/total
for i in range(n_classes0):
    fealist0_0[i]/=cntlist0[i]
    wlist0[i]=cntlist0[i]/total
flist.append(('ck_plus_48_feature.txt',O))

trainset = torchvision.datasets.CIFAR100(
    root='./data', train=True, download=True, transform=transform_train3)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=0)

total=0
O=torch.tensor([0.0] * featurelen)
n_classes1=100
fealist0_1 = [torch.tensor([0.0] * featurelen) for i in range(n_classes1)]
cntlist1 = [0] * n_classes1
wlist1=[0.0]*n_classes1
with torch.no_grad():
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs = inputs.expand(-1, 3, -1, -1)
        inputs, targets = inputs.to(device), targets.squeeze().long().to(device)
        tmplabels = targets.tolist()
        labellen = len(tmplabels)
        tmpfeature=net.forward_getfeature(inputs)
        O+=tmpfeature.sum(dim=0).cpu()
        total += targets.size(0)
        for i in range(labellen):
            # if (cntlist0[tmplabels[i]] == 100):
            #     continue
            cntlist1[tmplabels[i]] += 1
            fealist0_1[tmplabels[i]] += tmpfeature[i].cpu()
O=O/total
for i in range(n_classes1):
    fealist0_1[i]/=cntlist1[i]
    wlist1[i] = cntlist1[i] / total
flist.append(('cifar100_feature.txt',O))

print('==> Preparing data..')

data_transforms = [transforms.Resize((32,32))]
base_path_to_dataset='./data/FER_Plus/'

train_data = udata.FERplus(idx_set=0,
                               max_loaded_images_per_label=500000,
                               transforms=transforms.Compose(data_transforms),
                               base_path_to_FER_plus=base_path_to_dataset)

trainloader = DataLoader(train_data, batch_size=128, shuffle=True)

total=0
O=torch.tensor([0.0] * featurelen)
n_classes1=8
fealist0_1 = [torch.tensor([0.0] * featurelen) for i in range(n_classes1)]
cntlist1 = [0] * n_classes1
wlist1=[0.0]*n_classes1
with torch.no_grad():
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.squeeze().long().to(device)
        tmplabels = targets.tolist()
        labellen = len(tmplabels)
        tmpfeature=net.forward_getfeature(inputs)
        O+=tmpfeature.sum(dim=0).cpu()
        total += targets.size(0)
        for i in range(labellen):
            # if (cntlist0[tmplabels[i]] == 100):
            #     continue
            cntlist1[tmplabels[i]] += 1
            fealist0_1[tmplabels[i]] += tmpfeature[i].cpu()
O=O/total
for i in range(n_classes1):
    fealist0_1[i]/=cntlist1[i]
    wlist1[i] = cntlist1[i] / total
flist.append(('ferplus_feature.txt',O))

myfile=open('../../Player-Data/Input-P1-0','w')
for i in range(len(flist)):
    f=open('./AvgFeature-All/'+flist[i][0],'w')
    tmpo=flist[i][1].tolist()
    for j in range(featurelen):
        f.write(str(tmpo[j])+' ')
        myfile.write(str(tmpo[j])+' ')
    f.close()
myfile.close()