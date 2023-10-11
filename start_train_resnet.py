import os, argparse
import torch, torchvision
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision.datasets as datasets
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.models as models
import time, copy, numpy as np
from train_model import train_model
from test_model import test_model
from torchsummary import summary
import models
import csv
import sys
import datetime


parser = argparse.ArgumentParser(description='Tiny imagenet training')
parser.add_argument('--gpu', metavar='gpu', type=int, help='gpu number')
parser.add_argument('--conv', metavar='conv', type=str, default = 'static', help='conv type')
parser.add_argument('--dynamic', metavar='dynamic', type=int, default = 1, help='dynamic num')
parser.add_argument('--name', metavar='name', type=int, default = 1, help='name')
parser.add_argument('--epoch', metavar='epoch', type=int, default = 100, help='epoch num')
parser.add_argument('--batch', metavar='batch_size', type=int, default = 100, help='batch size')
args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.RandomAffine(40)], p=0.3),
        transforms.RandomApply([transforms.ColorJitter()], p=0.3),
        transforms.ToTensor(),
        transforms.Normalize([0.5070751592371323, 0.48654887331495095, 0.4409178433670343], [0.2673342858792401, 0.2564384629170883, 0.27615047132568404]),
    ]),
    'test': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5070751592371323, 0.48654887331495095, 0.4409178433670343], [0.2673342858792401, 0.2564384629170883, 0.27615047132568404]),
    ])
}

trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                        download=True, transform=data_transforms['train'])

testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                       download=True, transform=data_transforms['test'])

image_datasets = {'train' : trainset,
                  'test'  : testset}

dataloaders = {x: data.DataLoader(image_datasets[x], batch_size=args.batch, shuffle=True, num_workers=16)
                  for x in ['train', 'test']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}

model = models.ResNet18(conv_type = args.conv, dynamic=args.dynamic).cuda()
summary(model, input_size=(3, 64, 64))

criterion = nn.CrossEntropyLoss()
penalty = nn.L1Loss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay = 0.0001, nesterov = True)
exp_lr_scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

log_name = "ResNet18_"+args.conv+str(args.dynamic)+"_"+str(args.name)+"th_log"
with open('./'+log_name+'.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['epoch','time','loss', 'acc', 'val loss', 'val acc', 'best acc'])


model = train_model(model, dataloaders, dataset_sizes, criterion, optimizer, exp_lr_scheduler, penalty = penalty, dynamic = args.dynamic, conv_type = args.conv, log_name = log_name, num_epochs = args.epoch)
torch.save(model.module.state_dict(), "./models/ResNet18_"+args.conv+str(args.dynamic)+"_"+str(args.name)+"th.pth")

if os.path.exists('./Test_Acc.csv') == False:
    with open('./Test_Acc.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['name','loss','top1 acc', 'top5 acc'])

test_model(model, dataloaders, dataset_sizes, criterion, optimizer, name=log_name[:-4], conv_type = args.conv)
