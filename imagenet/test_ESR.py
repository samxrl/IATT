import torchvision.models as models
import torch
from PIL import Image
from torchvision import transforms
import os
from uitils.get_lable import get_lable
from tqdm import tqdm
import argparse


def getESR(modelName, iters):
    model1 = models.resnet50(pretrained=True)
    model2 = models.resnet101(pretrained=True)
    model3 = models.resnet152(pretrained=True)
    model4 = models.vgg16(pretrained=True)
    model5 = models.densenet161(pretrained=True)
    model6 = models.inception_v3(pretrained=True)
    model7 = models.vit_b_16(pretrained=True)

    model1.eval()
    model2.eval()
    model3.eval()
    model4.eval()
    model5.eval()
    model6.eval()
    model7.eval()

    if torch.cuda.is_available():
        model1.to('cuda')
        model2.to('cuda')
        model3.to('cuda')
        model4.to('cuda')
        model5.to('cuda')
        model6.to('cuda')
        model7.to('cuda')

    # Preprocess the image
    preprocess = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
    ])

    count1 = 0
    count2 = 0
    count3 = 0
    count4 = 0
    count5 = 0
    count6 = 0
    count7 = 0

    directory = 'samples/{}/transferable tests/iters_{}'.format(modelName, iters)

    org_directory = 'samples/{}/org'.format(modelName)
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
                output5 = model5(input_batch)
                output6 = model6(input_batch)
                output7 = model7(input_batch)

            _, predicted1 = torch.topk(output1.data, k=5)
            _, predicted2 = torch.topk(output2.data, k=5)
            _, predicted3 = torch.topk(output3.data, k=5)
            _, predicted4 = torch.topk(output4.data, k=5)
            _, predicted5 = torch.topk(output5.data, k=5)
            _, predicted6 = torch.topk(output6.data, k=5)
            _, predicted7 = torch.topk(output7.data, k=5)

            if predicted1[0][0].item() != predicted1[1][0].item():
                count1 += 1
            if predicted2[0][0].item() != predicted1[1][0].item():
                count2 += 1
            if predicted3[0][0].item() != predicted1[1][0].item():
                count3 += 1
            if predicted4[0][0].item() != predicted1[1][0].item():
                count4 += 1
            if predicted5[0][0].item() != predicted1[1][0].item():
                count5 += 1
            if predicted6[0][0].item() != predicted1[1][0].item():
                count6 += 1
            if predicted7[0][0].item() != predicted1[1][0].item():
                count7 += 1

            # print(predicted1.cpu().numpy(), predicted2.cpu().numpy(), filename)

        modelList = ['resnet50', 'resnet101', 'resnet152', 'vgg16', 'densenet161', 'inception_v3', 'vit_b_16']
        ESRs = [count1 / len(files), count2 / len(files), count3 / len(files), count4 / len(files), count5 / len(files),
                count6 / len(files), count7 / len(files)]

        for model, ESR in zip(modelList, ESRs):
            print('ESR on {}: {}'.format(model, ESR))


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
