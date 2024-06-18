import numpy as np
import os
import sys

sys.path.append("..")
from uitils.get_lable import get_lable
from uitils.gradCAM import getcam
import torch.nn as nn

import torchvision.models as models
from torchvision.models import Inception3, ResNet, DenseNet
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
import torchvision
import torch
import cv2
import argparse

def my_collate_fn(batch):
    images, targets = zip(*batch)
    images = torch.stack(images)
    return images, targets


def random_sample(modelName, k):
    data_dir = 'data'

    # Replace the original fully connected layer with a fully connected layer that has 91 output units.
    myModels = {'resnet50': models.resnet50(pretrained=True),
                'inception_v3': models.inception_v3(pretrained=True),
                'densenet161': models.densenet161(pretrained=True)}

    model = myModels[modelName]

    # 将原来的ResNet50的最后两层全连接层拿掉,替换成一个输出单元为10的全连接层
    if modelName in ['resnet50', 'inception_v3']:
        inchannel = model.fc.in_features
        model.fc = nn.Linear(inchannel, 90 + 1)
    else:  # densenet161
        inchannel = model.classifier.in_features
        model.classifier = nn.Linear(inchannel, 90 + 1)

    model.load_state_dict(torch.load('model/{}.ckpt'.format(modelName)))
    model.to(device)
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

    testset = CocoDetection(root=f'{data_dir}/val', annFile=f'{data_dir}/annotations/instances_val2017.json',
                            transform=preprocess)

    val_dataset_loader = DataLoader(testset, batch_size=1, shuffle=True, collate_fn=my_collate_fn, num_workers=1)

    count = 0
    for i, (images, targets) in enumerate(val_dataset_loader):

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

        labels = torch.tensor([t[0]['category_id'] if len(t) != 0 else 90 for t in targets], device=output.device)

        if labels[0].item() != predicted[0][0]:
            continue

        rgb_img = images[0].numpy()
        rgb_img = np.transpose(rgb_img, (1, 2, 0))
        rgb_img = (rgb_img - np.min(rgb_img)) / (np.max(rgb_img) - np.min(rgb_img))  # Convert to numpy to make pixel values greater than 1, and then normalize

        heatmap, visualization, orgImage, camArr = getcam(model, images, target_layer, labels[0], rgb_img)


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

        cv2.imwrite('samples/{}/org/{}-{}.jpg'.format(modelName, classIndex, i),
                    orgImage)
        cv2.imwrite(
            'samples/{}/visualization/{}-{}.jpg'.format(modelName, classIndex, i),
            visualization)
        cv2.imwrite('samples/{}/heatmap/{}-{}.jpg'.format(modelName, classIndex, i),
                    heatmap)

        np.save('samples/{}/CAMarr/{}-{}.npy'.format(modelName, classIndex, i), camArr)

        print('{}-{}'.format(classIndex, i))
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
