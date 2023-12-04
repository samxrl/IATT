from torch_dreams import Dreamer
import torch

from tqdm import tqdm
from copy import deepcopy

from torch_dreams.transforms import random_resize, pair_random_resize, pair_random_affine

from torch_dreams.auto_image_param import AutoImageParam
from torch_dreams.dreamer_utils import Hook, default_func_mean
from torch_dreams.masked_image_param import MaskedImageParam
from torch_dreams.batched_image_param import BatchedImageParam
from gradCAM import getcam
import numpy as np
import cv2
import os
from torchvision import transforms
from PIL import Image
from torchvision.models import Inception3, ResNet, DenseNet
import gc


class myDreamer(Dreamer):

    def render(
            self,
            model,
            saveFileName,
            layers,
            step=50,
            image_parameter=None,
            width=256,
            height=256,
            iters=120,
            lr=9e-3,
            rotate_degrees=15,
            scale_max=1.2,
            scale_min=0.5,
            translate_x=0.0,
            translate_y=0.0,
            custom_func=None,
            weight_decay=0.0,
            grad_clip=1.0,
    ):

        if image_parameter is None:
            image_parameter = AutoImageParam(
                height=height, width=width, device=self.device, standard_deviation=0.01
            )
        else:
            image_parameter = deepcopy(image_parameter)

        if image_parameter.optimizer is None:
            image_parameter.optimizer = image_parameter.fetch_optimizer(params_list=[image_parameter.param],
                                                                        optimizer=torch.optim.SGD, lr=lr,
                                                                        weight_decay=weight_decay)
            # image_parameter.get_optimizer(lr=lr, weight_decay=weight_decay)

        if self.transforms is None:
            self.get_default_transforms(
                rotate=rotate_degrees,
                scale_max=scale_max,
                scale_min=scale_min,
                translate_x=translate_x,
                translate_y=translate_y,
            )

        hooks = []
        for layer in layers:
            hook = Hook(layer)
            hooks.append(hook)

        if isinstance(image_parameter, MaskedImageParam):
            self.random_resize_pair = pair_random_resize(
                max_size_factor=scale_max, min_size_factor=scale_min
            )
            self.random_affine_pair = pair_random_affine(
                degrees=rotate_degrees, translate_x=translate_x, translate_y=translate_y
            )
        print('Generating{}'.format(saveFileName))
        for i in tqdm(range(iters), disable=self.quiet):

            image_parameter.optimizer.zero_grad()

            img = image_parameter.forward(device=self.device)

            if isinstance(image_parameter, MaskedImageParam):
                (
                    img_transformed,
                    mask_transformed,
                    original_image_transformed,
                ) = self.random_resize_pair(
                    tensors=[
                        img,
                        image_parameter.mask.to(self.device),
                        image_parameter.original_nchw_image_tensor,
                    ]
                )
                (
                    img_transformed,
                    mask_transformed,
                    original_image_transformed,
                ) = self.random_affine_pair(
                    [img_transformed, mask_transformed, original_image_transformed]
                )

                img = img_transformed * mask_transformed.to(self.device) + original_image_transformed.float() * (
                        1 - mask_transformed.to(self.device))

            else:
                img = self.transforms(img)

            model_out = self.model(img)

            layer_outputs = []

            for hook in hooks:
                ## if it's a BatchedImageParam, then include all batch items from hook output
                if isinstance(image_parameter, BatchedImageParam):
                    out = hook.output
                else:
                    ## else select only the first and only batch item
                    out = hook.output[0]

                layer_outputs.append(out)

            if custom_func is not None:
                loss = custom_func(layer_outputs)
            else:
                loss = self.default_func(layer_outputs)
            loss.backward()
            image_parameter.clip_grads(grad_clip=grad_clip)
            image_parameter.optimizer.step()

            lable = int(saveFileName.split('/')[-1].split('-')[0])

            if (i + 1) % step == 0:

                print('Generating iters_{} {}'.format(i + 1, saveFileName.replace('itersN', str(i + 1))))
                image_parameter.save(saveFileName.replace('itersN', str(i + 1)))

                if isinstance(image_parameter, MaskedImageParam):

                    if saveFileName.split('/')[1] == 'pull':
                        pull = True
                    else:
                        pull = False

                    if isinstance(model, Inception3):
                        target_layers = [model.Mixed_7c]
                        modelName = 'inception_v3'
                    elif isinstance(model, ResNet):
                        target_layers = [model.layer4[-1]]
                        modelName = 'resnet50'
                    elif isinstance(model, DenseNet):
                        target_layers = [model.features[-1]]
                        modelName = 'densenet161'

                    # Preprocess the image
                    preprocess = transforms.Compose([
                        transforms.ToTensor(),
                        # transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        #                      std=[0.229, 0.224, 0.225])
                    ])

                    images = Image.open(saveFileName.replace('itersN', str(i + 1)))
                    images = preprocess(images)
                    images = images.unsqueeze(0)  # create a mini-batch as expected by the model
                    images = images.to('cuda')

                    rgb_img = images[0].cpu().numpy()
                    rgb_img = np.transpose(rgb_img, (1, 2, 0))
                    rgb_img = (rgb_img - np.min(rgb_img)) / (
                            np.max(rgb_img) - np.min(rgb_img))  # 转numpy使像素值出现大于1的情况，归一化

                    heatmap, visualization, orgImage, camArr = getcam(model, images, target_layers, lable, rgb_img)
                    camArr = cv2.resize(camArr, (orgImage.shape[1], orgImage.shape[0]), interpolation=cv2.INTER_LINEAR)
                    camArr = (camArr - np.min(camArr)) / (np.max(camArr) - np.min(camArr) + 1e-7)
                    mask = torch.from_numpy(camArr)
                    mask = mask.unsqueeze(0).unsqueeze(0)
                    image_parameter.update_mask(mask)
                    if pull:
                        path = saveFileName.split('iters_')[0].replace(modelName, modelName + '/CAM')
                    else:
                        path = saveFileName.split('iters_')[0].replace(modelName, modelName + '/CAM')
                    path = path + saveFileName.split('/')[-1].split(',')[0]

                    if os.path.exists(path) is not True:
                        os.makedirs(path + '/heatmap')
                        os.makedirs(path + '/visualization')

                    visualization = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)
                    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)

                    cv2.imwrite(path + '/heatmap/iters_{}-'.format(i + 1) + saveFileName.split('/')[-1], heatmap)
                    cv2.imwrite(path + '/visualization/iters_{}-'.format(i + 1) + saveFileName.split('/')[-1],
                                visualization)

        for hook in hooks:
            hook.close()

        del heatmap, visualization, orgImage, camArr, images, rgb_img
        gc.collect()
        return image_parameter



