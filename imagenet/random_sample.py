import numpy as np
import os
import sys

sys.path.append("..")
from uitils.get_lable import get_lable
from uitils.gradCAM import getcam

import torchvision.models as models
from torchvision.models import Inception3, ResNet, DenseNet
from torchvision import transforms
from torch.utils.data import DataLoader
import torchvision
import torch
import cv2
import argparse


def random_sample(modelName, k):
    myModels = {'resnet50': models.resnet50(pretrained=True).to(device),
                'inception_v3': models.inception_v3(pretrained=True).to(device),
                'densenet161': models.densenet161(pretrained=True).to(device)}

    model = myModels[modelName]
    model.eval()

    if isinstance(model, Inception3):
        target_layer = [model.Mixed_7c]
    elif isinstance(model, ResNet):
        target_layer = [model.layer4[-1]]
    elif isinstance(model, DenseNet):
        target_layer = [model.features[-1]]

    preprocess = transforms.Compose([
        # transforms.Resize((224,224)),
        transforms.ToTensor(),
    ])

    val_dataset = torchvision.datasets.ImageFolder(
        root='data/val',
        transform=preprocess)

    val_dataset_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)

    count = 0
    for i, (images, labels) in enumerate(val_dataset_loader):

        # Use the following code to prevent video memory overflow due to oversized images
        size = images[0].shape
        if max(size[1], size[2]) > 1000:
            if size[1] > size[2]:
                resize = transforms.Compose([transforms.Resize((1000, int(size[2] / size[1] * 1000))), ])
            else:
                resize = transforms.Compose([transforms.Resize((int(size[1] / size[2] * 1000), 1000)), ])
            images = resize(images)

        # Classify the image
        with torch.no_grad():
            output = model(images.cuda())

        _, predicted = torch.topk(output.data, k=5)

        if labels[0].item() != predicted[0][0]:
            continue

        rgb_img = images[0].numpy()
        rgb_img = np.transpose(rgb_img, (1, 2, 0))
        rgb_img = (rgb_img - np.min(rgb_img)) / (np.max(rgb_img) - np.min(rgb_img))  # 转numpy使像素值出现大于1的情况，归一化

        heatmap, visualization, orgImage, camArr = getcam(model, images, target_layer, labels[0], rgb_img)

        lable_list = get_lable()
        imageNetLable = lable_list[labels[0].item()][0]
        className = lable_list[labels[0].item()][1]
        classIndex = labels[0].item()

        visualization = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)
        orgImage = cv2.cvtColor(np.uint8(255 * orgImage), cv2.COLOR_RGB2BGR)

        camArr = cv2.resize(camArr, (orgImage.shape[1], orgImage.shape[0]), interpolation=cv2.INTER_LINEAR)

        if os.path.exists('samples/{}/org'.format(modelName)) is not True:
            os.makedirs('samples/{}/org'.format(modelName))
        if os.path.exists('samples/{}/visualization'.format(modelName)) is not True:
            os.makedirs('samples/{}/visualization'.format(modelName))
        if os.path.exists('samples/{}/heatmap'.format(modelName)) is not True:
            os.makedirs('samples/{}/heatmap'.format(modelName))
        if os.path.exists('samples/{}/CAMarr'.format(modelName)) is not True:
            os.makedirs('samples/{}/CAMarr'.format(modelName))

        cv2.imwrite('samples/{}/org/{}-{}-{}-{}.jpg'.format(modelName, classIndex, i, imageNetLable, className),
                    orgImage)
        cv2.imwrite(
            'samples/{}/visualization/{}-{}-{}-{}.jpg'.format(modelName, classIndex, i, imageNetLable, className),
            visualization)
        cv2.imwrite('samples/{}/heatmap/{}-{}-{}-{}.jpg'.format(modelName, classIndex, i, imageNetLable, className),
                    heatmap)

        np.save('samples/{}/CAMarr/{}-{}-{}-{}.npy'.format(modelName, classIndex, i, imageNetLable, className), camArr)

        print('{}-{}-{}-{}'.format(classIndex, i, imageNetLable, className))
        count += 1
        if count >= k:
            break


if __name__ == '__main__':
    device = torch.device('cuda')
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-model",
                        default='resnet50',
                        type=str,
                        help="Name of WSM")
    parser.add_argument("--K", "-k",
                        default=1000,
                        type=int,
                        help="Number of original test inputs")

    args = parser.parse_args()

    modelName = args.model
    K = args.K

    if modelName not in ['resnet50', 'inception_v3', 'densenet161']:
        raise Exception("model mast in ['resnet50','inception_v3','densenet161']")

    random_sample(modelName, K)
