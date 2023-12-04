import numpy as np
from PIL import Image
from torchvision import transforms
import os
from IQA_pytorch import LPIPSvgg
from tqdm import tqdm
import argparse

preprocess = transforms.Compose([
    transforms.ToTensor(),
])

L = LPIPSvgg(channels=3).to('cuda')


def getLPIPS(modelName, iters):
    directory = 'samples/{}/transferable tests/iters_{}'.format(modelName, iters)
    org_directory = 'samples/{}/org'.format(modelName)

    res = np.empty(shape=(0), dtype=float)

    for root, dirs, files in os.walk(directory):
        if len(files) == 0:
            continue

        for filename in tqdm(files):
            input_image = Image.open(directory + "/{}".format(filename))
            input_tensor = preprocess(input_image)
            input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model
            input_batch = input_batch.to('cuda')

            org_image = Image.open(org_directory + "/{}".format(filename))
            org_tensor = preprocess(org_image)
            org_batch = org_tensor.unsqueeze(0)  # create a mini-batch as expected by the model
            org_batch = org_batch.to('cuda')

            LPIPS = L(input_batch, org_batch, as_loss=False)

            res = np.append(res, [LPIPS.item()], axis=0)

        res = np.mean(res)

        return res


if __name__ == "__main__":
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

    lpips = getLPIPS(modelName, iters)
    print('LPIPS: ' + str(lpips))
