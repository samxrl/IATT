import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
from datetime import datetime
import argparse
from torchvision.transforms import functional as F
import os
import cv2


def getTestAcc(model, epoch, test_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    correct = 0
    total = 0
    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            labels = torch.tensor([t[0]['category_id'] if len(t) != 0 else 91 for t in targets], device=outputs.device)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f'Epoch {epoch + 1}, Accuracy: {accuracy:.4f}')
    return accuracy


def my_collate_fn(batch):
    images, targets = zip(*batch)
    images = torch.stack(images)
    return images, targets


def train(modelName, num_epochs, batch_size, learning_rate):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_dir = 'data'
    model_dir = 'model'

    a = datetime.now()
    print('train start at : {}'.format(a))

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    trainset = CocoDetection(root=f'{data_dir}/train', annFile=f'{data_dir}/annotations/instances_train2017.json',
                             transform=transform)
    testset = CocoDetection(root=f'{data_dir}/val', annFile=f'{data_dir}/annotations/instances_val2017.json',
                            transform=test_transform)

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, collate_fn=my_collate_fn, num_workers=1)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, collate_fn=my_collate_fn, num_workers=1)

    myModels = {'resnet50': models.resnet50(pretrained=True),
                'inception_v3': models.inception_v3(pretrained=True),
                'densenet161': models.densenet161(pretrained=True)}

    model = myModels[modelName]

    # Replace the original fully connected layer with a fully connected layer that has 91 output units.
    if modelName in ['resnet50', 'inception_v3']:
        inchannel = model.fc.in_features
        model.fc = nn.Linear(inchannel, 90 + 1)
    else:  # densenet161
        inchannel = model.classifier.in_features
        model.classifier = nn.Linear(inchannel, 90 + 1)

    model.to(device)

    curr_lr = learning_rate

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    def update_lr(optimizer, lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    best_accuracy = 0.0
    total_step = len(train_loader)

    for epoch in range(num_epochs):
        model.train()
        for i, (images, targets) in enumerate(train_loader):

            images = images.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            if modelName == 'inception_v3':
                outputs = outputs.logits

            labels = torch.tensor([t[0]['category_id'] if len(t) != 0 else 90 for t in targets], device=outputs.device)

            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            if (i + 1) % 10 == 0:
                print("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}"
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

            if (epoch + 1) % 3 == 0:
                curr_lr /= 10
                update_lr(optimizer, curr_lr)

        model.eval()
        accuracy = getTestAcc(model, epoch, test_loader)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            bestepoch = epoch + 1
            torch.save(model.state_dict(), '{}/{}.ckpt'.format(model_dir, modelName))
            print('bestacc: {}, bestepoch: {}'.format(best_accuracy, bestepoch))

    b = datetime.now()
    print('train end at : {}'.format(b))
    print('train cost : {}'.format(b - a))


if __name__ == '__main__':
    print(torch.cuda.is_available())
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-model",
                        default='resnet50',
                        type=str,
                        help="Name of WSM")
    parser.add_argument("--epochs", "-epochs",
                        default=50,
                        type=int,
                        help="num_epochs")
    parser.add_argument("--batch", "-batch",
                        default=50,
                        type=int,
                        help="batch_size")
    parser.add_argument("--lr", "-lr",
                        default=0.001,
                        type=float,
                        help="learning_rate")

    args = parser.parse_args()

    modelName = args.model
    num_epochs = args.epochs
    batch_size = args.batch
    learning_rate = args.lr

    if modelName not in ['resnet50', 'inception_v3', 'densenet161']:
        raise Exception("model mast in ['resnet50','inception_v3','densenet161']")

    train(modelName, num_epochs, batch_size, learning_rate)
