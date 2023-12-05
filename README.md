# IATT

IATT is an Interpretation Analysis based Transferable Test Generation Method for Convolutional Neural Networks. The workflow of IATT is illustrated in the following figure:

![The Workflow of IATT](/images/workflow.svg "The Workflow of IATT")

## Install

We suggest installing IATT in `Python 3.8` or higher versions, and run it on a CUDA GPU with at least 8G VRAM.You can use the following command to install all the packages required by IATT.

```bash
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
pip install -r requirements.txt
```

## Usage

Due to the different handling of models and test inputs, we have placed the code for `ImageNet` and `CIFAR-10` dataset in two separate directories. Unless otherwise specified, `.py` files with the same file name in both directories have the same functionality and share the same parameters.

### 1. Dataset format

#### for ImageNet

We sample original test inputs from the validation set of ImageNet2012 (ILSVRC2012). You can download our formatted validation set [here](https://drive.google.com/file/d/1As-8IfRNbcQjAR3ev7GIHFtv2bzclXal/view?usp=sharing), or the complete official dataset [here](https://image-net.org/challenges/LSVRC/2012/2012-downloads.php). Then, place the validation set in the `imagenet` directory following this directory structure:

```bash
imagenet     
    └─data
       ├─synset_words.txt # imageNet label ID and description correspondence (already exists)                   
       └─val                   
           ├─n01440764 # The directory storing images with the label ID n01440764
           │  ├─ILSVRC2012_val_00000293.JPEG
           │  ├─ILSVRC2012_val_00002138.JPEG
           │  ├─......       
           ├─n01443537   
           ├─......
```

#### for CIFAR-10

We use `torchvision` to download and load the `CIFAR-10` dataset. Upon first download, a directory named `data` will be created in the project's root directory to store the dataset. No additional processing is required.

### 2. Get models

In IATT, transferable test inputs are generated by analyzing the internal information of the White-box Source Model (WSM) and use them to test the Black-box Target Model (BTM) with the similar function to WSM.

#### for ImageNet

we utilized the pre-trained models provided by [PyTorch](https://pytorch.org/vision/stable/models.html) as experimental models. including `ResNet-50`, `DenseNet-161`, and `Inception V3`, as the WSMs, and chose the above three models and `ResNet-101`, `ResNet-152`, `VGG16`, `Vision Transformer-B/16` as BTMs, which have similar function to WSM. **All pre-trained models will be automatically downloaded through our code.**

#### for CIFAR-10

we chose `ResNet-50` as WSM out from the ImageNet image classification models, and `VGG16`, `DenseNet-161`, `Inception V3` as BTMs. To classify the images in the CIFAR-10 dataset with the above models, we performed transfer learning on the four pre-trained models using the CIFAR-10 dataset.

You can download the models trained in our experiments [here](https://drive.google.com/drive/folders/1GVcJGUl02UR8p-YVYCJ9Q9xZKlz0liKG?usp=sharing) and put them in `cifar10/model`, or run `train.py` to train the models yourself:

```bash
python train.py --model <Name of WSM> --epochs <Number of epochs> --batch <batch_size> --lr <learning_rate>
```

**parameters**

* `model`: Name of WSM. Only supports `resnet50`(default), `inception_v3`, `densenet161`, or `vgg16`.
* `epochs`,`batch`,`lr`: Hyperparameters
  * To achieve high accuracy, we recommend setting the following hyperparameters:
    * `epochs` : 50
    * `batch` : 50
    * `lr` : 0.001

### 3. Sample original test input

For each dataset, we randomly selected images that had been correctly classified by WSM as the original test inputs.

Run `random_sample.py` to randomly sample original test inputs from the test set of the dataset:

```bash
python random_sample.py --model <Name of WSM> --K <Number of original test inputs>
```

**parameters**

* `model` : Name of WSM. Only supports `resnet50`(default), `inception_v3`, or `densenet161`.
* `K` : The number of original test inputs, defaulting to `1000`

For each dataset, the following directories are automatically created to store the sampled original test inputs and
their interpretation analysis results (CAMs):

```bash
<dataset>
  └─samples
        └─<Name of WSM>
            ├─CAMarr # .npy files, stored in the format of numpy arrays for CAMs
            ├─heatmap # .jpg files, CAM visualization (heat map)
            ├─org # .jpg files, sampled original test input images
            └─visualization # .jpg files, Visualization of the CAM superimposed on the original test inputs
```

### 4. Generate Transferable Tests

IATT generat transferable test inputs by analyzing the internal information of the WSM and use them to test the BTMs with the similar function to WSM.

Run `run.py` to generate transferable test inputs:

```bash
python run.py --model <Name of WSM> --iters <Number of iterations> --step <Iteration interval for each CAM update>
```

**parameters**

* `model` : Name of WSM. Only supports `resnet50`(default), `inception_v3`, or `densenet161`.
* `iters` : Number of iterations, defaulting to `300`
* `step` : Iteration interval for each CAM update, defaulting to `20`

For each dataset, the following directories are automatically created to store the generated transferable test inputs and their interpretation analysis results generated in each `step` of iteration：

```bash
<dataset>
  └─samples
        └─<Name of WSM>
            ├─transferable tests
            │  ├─iters_<step>
            │  │  ├─<lable ID>-<image ID>-<lable description>.jpg # generated transferable test
            │  │  ├─......
            │  ├─......
            └─CAM # Storing interpretation analysis results at each step of iteration
               ├─<lable ID>-<image ID>-<lable description> # CAM of same-name transferable test inputs
               │  ├─heatmap
               │  │  ├─iters_<step>-<lable ID>-<image ID>-<lable description>.jpg 
               │  │  ├─......
               │  └─visualization
               │     ├─iters_<step>-<lable ID>-<image ID>-<lable description>.jpg
               │     ├─......
               ├─......
```

The generated transferable test inputs will maintain the same size as the original test inputs.

## Verify

To evaluate the transferability of the test inputs generated by IATT, we utilize two metrics, Error-inducing Success Rate (ESR) and Authenticity(LPIPS).

### ESR

Run `test_ESR.py` to use the generated transferable test inputs to test multiple different BMTs, and calculate ESR:
```bash
python test_ESR.py --model <Name of WSM> --iters <Number of iterations>
```

**parameters**
* `model` : WSM of generated transferable test inputs. Only supports `resnet50`(default), `inception_v3`, or `densenet161`.
* `iters` : Transferable test inputs generated on which iteration, defaulting to `300`

The ESR of test each BTM with transferable test inputs will be printed in the terminal.

### LPIPS

Run `test_LPIPS.py` to evaluate the authenticity of the transferable test inputs by calculating the LPIPS between the generated transferable test inputs and the original test inputs：
```bash
python test_LPIPS.py --model <Name of WSM> --iters <Number of iterations>
```

**parameters**
* `model` : WSM of generated transferable test inputs. Only supports `resnet50`(default), `inception_v3`, or `densenet161`.
* `iters` : Transferable test inputs generated on which iteration, defaulting to `300`

The average LPIPS of transferable test inputs will be printed in the terminal.
