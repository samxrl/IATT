import argparse
import cv2
import gc
import numpy as np
import os
import random
import shutil
import torch
import torchvision
import torchvision.models as models
from torch.utils.data import DataLoader
from torch_dreams.masked_image_param import MaskedImageParam
from torchvision import transforms
from torchvision.models import Inception3, ResNet, DenseNet

from uitils.get_lable import get_lable
from uitils.gradCAM import getcam
from uitils.myDreamer import myDreamer


def generateTransferableTest(modelName, iters=300, step=20):
    myModels = {'resnet50': models.resnet50(pretrained=True).to(device),
                'inception_v3': models.inception_v3(pretrained=True).to(device),
                'densenet161': models.densenet161(pretrained=True).to(device)}

    model = myModels[modelName]
    model.eval()

    def make_custom_func(classN):
        def custom_func(layer_outputs):
            loss = -layer_outputs[0][classN]
            return -loss

        return custom_func

    def featureVisualization(layers_to_use, iters, step, classN, img, mask, dreamy_boi, saveFileName, model):
        my_custom_func = make_custom_func(classN)
        param = MaskedImageParam(image=img, mask_tensor=mask, device='cuda')

        image_param = dreamy_boi.render(
            model=model,
            saveFileName=saveFileName,
            image_parameter=param,
            layers=layers_to_use,
            custom_func=my_custom_func,
            iters=iters,
            step=step,
            width=224,
            height=224,
            lr=2e-5,
            grad_clip=0.1,
        )

        return image_param

    dreamy_boi = myDreamer(model, device='cuda')

    if modelName in ['resnet50', 'inception_v3']:
        layers_to_use = [model.fc]
    else:  # densenet161
        layers_to_use = [model.classifier]

    directory = 'samples/{}/org'.format(modelName)
    for root, dirs, files in os.walk(directory):
        if len(files) == 0:
            raise Exception("Please run random_sample.py first to sample the original image")

        for i, filename in enumerate(files):
            image = root + '/' + filename
            CAMarr = np.load('samples/{}/CAMarr/'.format(modelName) + filename.replace('.jpg', '.npy'))

            classN = int(filename.split('-')[0])

            print('Generating iters_{} {}.jpg'.format(iters, filename.split('.')[0]))

            for i in range(step, iters + step, step):
                if os.path.exists(
                        'samples/{}/transferable tests/iters_{}'.format(modelName, i)) is not True:
                    os.makedirs('samples/{}/transferable tests/iters_{}'.format(modelName, i))

            if found('samples/{}/transferable tests/iters_{}'.format(modelName, iters), filename):
                print('exist {}'.format(filename))
                continue

            mask = makeMask(CAMarr)

            saveFileName = 'samples/{}/transferable tests/iters_itersN/{}.jpg'.format(modelName,
                                                                                      filename.split('.jp')[0])
            try:
                featureVisualization(layers_to_use, iters, step, classN, image, mask, dreamy_boi,
                                     saveFileName, model)
            except:
                raise Exception("Image size too large or non-RGB image")


def makeMask(CAMarr):
    CAMarr = normalization(CAMarr)
    mask = torch.from_numpy(CAMarr)
    mask = mask.unsqueeze(0).unsqueeze(0)
    return mask


def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


def found(imagePath, filename):
    imgs = os.listdir(imagePath)
    for f in imgs:
        if filename.split('.')[0] in f and f.endswith('.jpg'):
            return True
    return False


if __name__ == "__main__":
    device = torch.device('cuda')

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-model",
                        default='resnet50',
                        type=str,
                        help="Name of WSM")
    parser.add_argument("--iters", "-iters",
                        default=300,
                        type=int,
                        help="Number of iterations")

    parser.add_argument("--step", "-step",
                        default=20,
                        type=int,
                        help="Iteration interval for each CAM update")

    args = parser.parse_args()

    modelName = args.model
    iters = args.iters
    step = args.step

    if modelName not in ['resnet50', 'inception_v3', 'densenet161']:
        raise Exception("model mast in ['resnet50','inception_v3','densenet161']")

    generateTransferableTest(modelName, iters, step)
