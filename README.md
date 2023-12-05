# IATT

IATT is an Interpretation Analysis based Transferable Test Generation Method for Convolutional Neural Networks. The
workflow of
IATT is illustrated in the following figure:

![The Workflow of IATT](/images/workflow.svg "The Workflow of IATT")

## Install

Suggest installing IATT in `Python 3.8` or higher versions, and run it on a CUDA GPU with at least 8G VRAM.You can use
the following command to install all the packages required by IATT.

```bash
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
pip install -r requirements.txt
```

## Usage

Due to the different handling of models and test inputs, we have placed the code for `ImageNet` and `CIFAR-10` dataset
in two
separate directories. Unless otherwise specified, `.py` files with the same file name in both directories have the same
functionality and share the same parameters.

### 1. Get models

In IATT, transferable test inputs are generated by analyzing the internal information of the White-box Source Model (
WSM)
and use them to test the Black-box Target Model (BTM) with the similar function to WSM.

#### for ImageNet

we utilized the pre-trained models provided by [PyTorch](https://pytorch.org/vision/stable/models.html)
as experimental models. including `ResNet-50`, `DenseNet-161`, and `Inception V3`, as the WSMs, and chose the above
three
models and `ResNet-101`, `ResNet-152`, `VGG16`, `Vision Transformer-B/16` as BTMs, which have similar function to
WSM. **All
pre-trained models will be automatically downloaded through our code.**

#### for CIFAR-10

we chose `ResNet-50` as WSM out from the ImageNet image classification
models, and `VGG16`, `DenseNet-161`, `Inception V3` as BTMs. To classify the images in the CIFAR-10 dataset with the
above models, we performed transfer learning on the four pre-trained models using the CIFAR-10
dataset.

You can download the models trained in our
experiments [here](https://drive.google.com/drive/folders/1GVcJGUl02UR8p-YVYCJ9Q9xZKlz0liKG?usp=sharing) and put them
in `cifar10/model`, or run
`train.py` to train the models yourself:

```bash
python train.py --model <Name of WSM> --epochs <Number of epochs> --batch <batch_size> --lr <learning_rate>
```

**parameters**

* `model`: Name of WSM. Only supports `resnet50`(default), `inception_v3`, or `densenet161`.
* `epochs`,`batch`,`lr`: Hyperparameters
    * To achieve high accuracy, we recommend setting the following hyperparameters:
        * `epochs` : 50
        * `batch`: 50
        * `lr`: 0.001

### 2. Sample original test input

For each dataset, we randomly selected images that had been correctly
classified by WSM as the original test inputs to generate test inputs.

Run `random_sample.py` to randomly sample original test inputs from the test set of the dataset:

```bash
python random_sample.py --model <Name of WSM> --K <Number of original test inputs>
```

**parameters**

* `model`: Name of WSM. Only supports `resnet50`(default), `inception_v3`, or `densenet161`.
* `K`: The number of original test inputs, defaulting to `1000`

For each dataset, the following directories are automatically created to store the sampled original test inputs and their
interpretation analysis results (CAMs):

```bash
samples
└─<Name of WSM>
    ├─CAMarr # .npy files, stored in the format of numpy arrays for CAMs
    ├─heatmap # .jpg files, CAM visualization (heat map)
    ├─org # .jpg files, original test input images
    └─visualization # .jpg files, Visualization of the CAM superimposed on the original test inputs
```

### 3. Generate Transferable Tests




