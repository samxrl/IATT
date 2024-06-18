import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision import models
from datetime import datetime
import argparse
import os


def getTestAcc(model, epoch, test_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Accuracy of the model on the test images: {} % In epoch {} '.format((100 * correct / total), epoch))
    return 100 * correct / total


def train(modelName, num_epochs, batch_size, learning_rate):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
        transforms.Pad(4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32),
        transforms.ToTensor(),
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC)
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC)
    ])

    num_classes = 10
    train_dataset = torchvision.datasets.SVHN(root='../data/',
                                              split='train',
                                              transform=transform,
                                              download=True)

    test_dataset = torchvision.datasets.SVHN(root='../data/',
                                             split='test',
                                             download=True,
                                             transform=test_transform)

    # 数据载入
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)

    # 定义模型
    myModels = {'resnet50': models.resnet50(pretrained=True),
                'inception_v3': models.inception_v3(pretrained=True),
                'densenet161': models.densenet161(pretrained=True),
                'vgg16': models.vgg16(pretrained=True)}

    model = myModels[modelName]

    # The original ResNet50's last two fully connected layers are removed and replaced with a fully connected layer with 10 output units
    if modelName in ['resnet50', 'inception_v3']:
        inchannel = model.fc.in_features
        model.fc = nn.Linear(inchannel, 10)
    elif modelName == 'vgg16':
        inchannel = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(inchannel, 10)
    else:  # densenet161
        inchannel = model.classifier.in_features
        model.classifier = nn.Linear(inchannel, 10)

    model.to(device)
    model.train()

    # loss
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    a = datetime.now()

    print('train start at : {}'.format(a))

    # 训练数据集
    total_step = len(train_loader)
    curr_lr = learning_rate
    bestacc = 0
    bestepoch = 0

    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):

            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)

            if modelName == 'inception_v3':
                outputs = outputs.logits
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 30 == 0:
                print("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}"
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

        model.eval()
        temp = getTestAcc(model, epoch + 1, test_loader=test_loader)
        model.train()
        if temp > bestacc:
            bestacc = temp
            bestepoch = epoch + 1

            if os.path.exists('model'.format(modelName)) is not True:
                os.makedirs('model'.format(modelName))

            torch.save(model.state_dict(), 'model/{}.ckpt'.format(modelName))

            print('Model Successfully Saved')

    print('bestacc: {}, bestepoch: {}'.format(bestacc, bestepoch))

    # 测试网络模型
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Accuracy of the model on the test images: {} %'.format(100 * correct / total))

    b = datetime.now()
    print('train end at : {}'.format(b))
    print('train cost : {}'.format(b - a))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-model",
                        default='densenet161',
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

    if modelName not in ['resnet50', 'inception_v3', 'densenet161', 'vgg16']:
        raise Exception("model mast in ['resnet50','inception_v3','densenet161','vgg16']")

    train(modelName, num_epochs, batch_size, learning_rate)
