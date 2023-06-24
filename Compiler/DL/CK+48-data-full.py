import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from CK_Plus_DataSet import *
import os
import random
def set_seed(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

set_seed(0)

transform_train = transforms.Compose([
    transforms.Resize((32, 32)),
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
trainset =CK(split = 'Training', fold = 1, transform=transform_train,path='./data/CK_data.h5')
trainloader = DataLoader(trainset,batch_size=1,shuffle=False,drop_last=False)

testset = CK(split = 'Testing', fold = 1, transform=transform_test,path='./data/CK_data.h5')
testloader = DataLoader(testset,batch_size=1,shuffle=False,drop_last=False,)
myfile=open('../../Player-Data/Input-P0-0','w')

num_class=7
imglist=[]
targetlist=[]

def label2txt(mylist):
    for i in range(len(mylist)):
        myfile.write(str(mylist[i])+' ')
    myfile.write('\n')
    myfile.flush()

def img2txt(mylist):
    for i in range(0,len(mylist),3*32*32):
        for j in range(32*32):
            for k in range(3):
                myfile.write(str(mylist[i+1024*k+j])+' ')
        myfile.write('\n')
    myfile.flush()

total0 = 0
for batch_idx, (inputs, targets) in enumerate(trainloader):
    inputs = inputs.expand(-1, 3, -1, -1).resize((3*32*32)).tolist()
    imglist+=inputs
    tmplist=[0]*num_class
    tmplist[targets.tolist()[0]]=1
    targetlist+=tmplist
    total0 += targets.size(0)

print(total0)
label2txt(targetlist)
img2txt(imglist)

imglist=[]
targetlist=[]
total1 = 0
for batch_idx, (inputs, targets) in enumerate(testloader):
    inputs = inputs.expand(-1, 3, -1, -1).resize((3 * 32 * 32)).tolist()
    imglist += inputs
    tmplist = [0] * num_class
    tmplist[targets.tolist()[0]] = 1
    targetlist += tmplist
    total1 += targets.size(0)

print(total1)
label2txt(targetlist)
img2txt(imglist)

myfile.flush()
myfile.close()
