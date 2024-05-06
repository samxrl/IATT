import numpy as np
import os
import sys
import torch.nn as nn

sys.path.append("..")
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
    myModels = {'resnet50': models.resnet50(pretrained=True),
                'inception_v3': models.inception_v3(pretrained=True),
                'densenet161': models.densenet161(pretrained=True)}

    model = myModels[modelName]

    # The original ResNet50's last two fully connected layers are removed and replaced with a fully connected layer with 10 output units
    if modelName in ['resnet50', 'inception_v3']:
        inchannel = model.fc.in_features
        model.fc = nn.Linear(inchannel, 10)
    else:  # densenet161
        inchannel = model.classifier.in_features
        model.classifier = nn.Linear(inchannel, 10)

    model.load_state_dict(torch.load('model/{}.ckpt'.format(modelName)))
    model.to(device)
    model.eval()

    # Image Preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC)
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC)
    ])

    test_dataset = torchvision.datasets.SVHN(root='../data/',
                                                split='test',
                                                transform=test_transform)

    val_dataset_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    if isinstance(model, Inception3):
        target_layer = [model.Mixed_7c]
    elif isinstance(model, ResNet):
        target_layer = [model.layer4[-1]]
    elif isinstance(model, DenseNet):
        target_layer = [model.features[-1]]

    count = 0
    for i, (images, labels) in enumerate(val_dataset_loader):

        # Classify the image
        with torch.no_grad():
            output = model(images.cuda())

        _, predicted = torch.topk(output.data, k=5)

        if labels[0].item() != predicted[0][0]:
            continue

        rgb_img = images[0].numpy()
        rgb_img = np.transpose(rgb_img, (1, 2, 0))
        rgb_img = (rgb_img - np.min(rgb_img)) / (np.max(rgb_img) - np.min(rgb_img))  # Convert to numpy to make pixel values greater than 1, and then normalize

        heatmap, visualization, orgImage, camArr = getcam(model, images, target_layer, labels[0], rgb_img)

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

        cv2.imwrite('samples/{}/org/{}-{}.jpg'.format(modelName, labels[0].item(), i), orgImage)
        cv2.imwrite('samples/{}/visualization/{}-{}.jpg'.format(modelName, labels[0].item(), i),
                    visualization)
        cv2.imwrite('samples/{}/heatmap/{}-{}.jpg'.format(modelName, labels[0].item(), i),
                    heatmap)

        np.save('samples/{}/CAMarr/{}-{}.npy'.format(modelName, labels[0].item(), i), camArr)

        print('{}-{}'.format(labels[0].item(), i))
        count += 1
        if count >= k:
            break


if __name__ == '__main__':
    device = torch.device('cuda')
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-model",
                        default='inception_v3',
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
