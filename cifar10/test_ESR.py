import torchvision.models as models
import torch
from PIL import Image
import torch.nn as nn
from torchvision import transforms
import os
from tqdm import tqdm
import argparse
import numpy as np

def getModel(modelName):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    myModels = {'resnet50': models.resnet50(pretrained=True),
                'vgg16': models.vgg16(pretrained=True),
                'inception_v3': models.inception_v3(pretrained=True),
                'densenet161': models.densenet161(pretrained=True)}

    model = myModels[modelName]

    # Replace the original fully connected layer with a fully connected layer that has 10 output units.
    if modelName in ['resnet50', 'inception_v3']:
        inchannel = model.fc.in_features
        model.fc = nn.Linear(inchannel, 10)
    elif modelName == 'vgg16':
        inchannel = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(inchannel, 10)
    else:  # densenet161
        inchannel = model.classifier.in_features
        model.classifier = nn.Linear(inchannel, 10)

    model.load_state_dict(torch.load('model/{}.ckpt'.format(modelName)))
    return model.to(device)


def getESR(modelName, iters):
    model1 = getModel('resnet50')
    model2 = getModel('vgg16')
    model3 = getModel('inception_v3')
    model4 = getModel('densenet161')

    model1.eval()
    model2.eval()
    model3.eval()
    model4.eval()


    if torch.cuda.is_available():
        model1.to('cuda')
        model2.to('cuda')
        model3.to('cuda')
        model4.to('cuda')

    # Preprocess the image
    preprocess = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
    ])

    count1 = 0
    count2 = 0
    count3 = 0
    count4 = 0


    directory = 'samples/{}/transferable tests/iters_{}'.format(modelName, iters)

    org_directory = 'samples/{}/org'.format(modelName)
    lables = np.zeros(shape=(4, 10), dtype=int)
    for root, dirs, files in os.walk(directory):
        if len(files) == 0:
            continue

        for i, filename in enumerate(tqdm(files)):

            input_image = Image.open(directory + "/{}".format(filename))

            if len(input_image.split()) != 3:
                input_image = input_image.convert('RGB')

            input_tensor = preprocess(input_image)

            if input_tensor.shape[0] == 1:
                input_tensor = torch.stack((input_tensor[0], input_tensor[0], input_tensor[0]), dim=0)

            input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model
            input_batch = input_batch.to('cuda')

            org_image = Image.open(org_directory + "/{}".format(filename))
            org_tensor = preprocess(org_image)
            org_batch = org_tensor.unsqueeze(0)  # create a mini-batch as expected by the model
            org_batch = org_batch.to('cuda')

            input_batch = torch.cat([input_batch, org_batch], dim=0)

            # Classify the image
            with torch.no_grad():
                output1 = model1(input_batch)
                output2 = model2(input_batch)
                output3 = model3(input_batch)
                output4 = model4(input_batch)

            _, predicted1 = torch.topk(output1.data, k=5)
            _, predicted2 = torch.topk(output2.data, k=5)
            _, predicted3 = torch.topk(output3.data, k=5)
            _, predicted4 = torch.topk(output4.data, k=5)

            if predicted1[0][0].item() != predicted1[1][0].item():
                count1 += 1
                lables[0, predicted1[0][0].item()] = 1
            if predicted2[0][0].item() != predicted2[1][0].item():
                count2 += 1
                lables[1, predicted2[0][0].item()] = 1
            if predicted3[0][0].item() != predicted3[1][0].item():
                count3 += 1
                lables[2, predicted3[0][0].item()] = 1
            if predicted4[0][0].item() != predicted4[1][0].item():
                count4 += 1
                lables[3, predicted4[0][0].item()] = 1

        modelList = ['resnet50', 'vgg16', 'inception_v3', 'densenet161']
        ESRs = [count1 / len(files), count2 / len(files), count3 / len(files), count4 / len(files)]

        for model, ESR in zip(modelList, ESRs):
            print('ESR on {}: {}'.format(model, ESR))
        print('--------------------------------------')
        for model, lableNum in zip(modelList[:7], np.sum(lables, axis=1)):
            print('lableNum of {}: {}'.format(model, lableNum))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-model",
                        default='resnet50',
                        type=str,
                        help="Name of WSM")
    parser.add_argument("--iters", "-iters",
                        default=300,
                        type=int,
                        help="Number of iterations of the samples to test")

    args = parser.parse_args()

    modelName = args.model
    iters = args.iters

    if modelName not in ['resnet50', 'inception_v3', 'densenet161']:
        raise Exception("model mast in ['resnet50','inception_v3','densenet161']")

    getESR(modelName, iters)
