import torch, time, sys, csv

def test_model(model, dataloaders, dataset_sizes, criterion, optimizer, name, conv_type):
    tik = time.time()

    for phase in ['test']:
        model.eval()

        running_loss = 0.0
        running_corrects = 0
        top_correct_5 = 0.0

        for i,(inputs, labels) in enumerate(dataloaders[phase]):
            inputs = inputs.cuda()
            labels = labels.cuda()

            optimizer.zero_grad()

            with torch.set_grad_enabled(False):

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                _, top_pred = outputs.topk(5, 1, largest=True, sorted=True)

                top_label = labels.view(labels.size(0), -1).expand_as(top_pred)
                top_correct = top_pred.eq(top_label).float()
                top_correct_5 += top_correct[:, :5].sum()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            print("\rIteration: {}/{}, Loss: {}.".format(i+1, len(dataloaders[phase]), loss.item() * inputs.size(0)), end="")
            sys.stdout.flush()

        epoch_top5 = top_correct_5 / dataset_sizes[phase]
        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_acc = running_corrects.double() / dataset_sizes[phase]
    print()
    print('Test Loss: {:.4f} Acc: {:.4f} Top5 Acc: {:.4f}'.format(epoch_loss, epoch_acc, epoch_top5))
    print()


    tok = time.time() - tik
    print('Test complete in {:.0f}m {:.0f}s'.format(
        tok // 60, tok % 60))

    with open('./Test_Acc.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow([name,str(epoch_loss),str(epoch_acc.item()), str(epoch_top5.item())])
