import torch, torchvision
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision.datasets as datasets
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.models as models
import matplotlib.pyplot as plt
import time, os, copy, numpy as np
import sys
import datetime
import csv

def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, penalty, dynamic, conv_type, log_name, num_epochs=25):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_acc5 = 0.0


    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('Current lr : ' + str(scheduler.get_last_lr()[0]))
        print('-' * 10)

        for phase in ['train', 'test']:
            if phase == 'train':
                if epoch == 0:
                    pass
                else:
                    scheduler.step()
                model.train() 
            else:
                model.eval()  

            running_loss = 0.0
            running_corrects = 0
            top_correct_5 = 0.0

            for i,(inputs, labels) in enumerate(dataloaders[phase]):
                inputs = inputs.cuda()
                labels = labels.cuda()

                optimizer.zero_grad()

                if phase == 'train':
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    
                else:
                    with torch.no_grad():
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                _, top_pred = outputs.topk(5, 1, largest=True, sorted=True)

                top_label = labels.view(labels.size(0), -1).expand_as(top_pred)
                top_correct = top_pred.eq(top_label).float()
                top_correct_5 += top_correct[:, :5].sum()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                now = datetime.datetime.now()
                print("\r" + now.strftime('%Y-%m-%d %H:%M:%S') + ", Iteration: {}/{}, Loss: {}".format(i+1, len(dataloaders[phase]), loss.item() * inputs.size(0)), end="")
                
            epoch_top5 = top_correct_5 / dataset_sizes[phase]
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            if phase == 'train':
                avg_loss = epoch_loss
                t_acc = epoch_acc
            else:
                val_loss = epoch_loss
                val_acc = epoch_acc
            
            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_acc5 = epoch_top5
                best_model_wts = copy.deepcopy(model.state_dict())


        now = datetime.datetime.now()
        with open('./'+log_name+'.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow([epoch,now.strftime('%Y-%m-%d %H:%M:%S'),str(avg_loss),str(t_acc.item()),str(val_loss),str(val_acc.item()),str(best_acc.item())])

        print()
        print('Train Loss: {:.4f} Acc: {:.4f}'.format(avg_loss, t_acc))
        print('Test Loss: {:.4f} Acc: {:.4f} Top5 Acc: {:.4f}'.format(val_loss, val_acc, epoch_top5))
        print('Best Test Accuracy: {}  Top5 Accuracy: {}'.format(best_acc, best_acc5))
        print()
    
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best test Acc: {:4f} Top5 Acc: {:.4f}'.format(best_acc, epoch_top5))

    model.load_state_dict(best_model_wts)
    return model