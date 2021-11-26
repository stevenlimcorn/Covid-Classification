# Covid-Classification

### Introduction

This is an image classification model that classifies chest x-ray images as lung opacity, normal, viral pneumonia or COVID-19 positive. The data is retrieved from kaggle, created by [tawsifurrahman](https://www.kaggle.com/tawsifurrahman/covid19-radiography-database). The dataset used has 6012, 10192, 1345, 3616 number of images for lung opacity, normal, viral pneumonia, COVID-19 chest x-ray images respectively. This is a ViT based model which is pretrained on Imagenet dataset, with a specification of image size 224 and 16 patches. The model is transfer learn to match the dataset given by [tawsifurrahman](https://www.kaggle.com/tawsifurrahman/covid19-radiography-database). The overall test accuracy is 95.8%, with a test loss of 0.1243.

### Code

Libraries Needed
```python
import torch
import random
import os
import numpy as np
import torch.nn as nn
import pandas as pd
import math
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sn
from torch.utils.data import DataLoader
from PIL import Image
import timm
import torchvision.transforms as transforms
import glob
```

### Load Data
The data is split into 3 dataset which includes training, validation and test data, it is split into 80%, 10%, 10% respectively.

#### Data Augmentation
```python
transforms_train = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomResizedCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)

transforms_test = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)
```

Chest x-ray has no difference between left and right, as a result augmenting data, flipping horizontally would be an ideal augmentation. The result shows significant improvement. To make sure all images have the same size, the image is resized to a specific `IMG_SIZE` of 224. The Normalization method are just mean and std of imagenet data, since I have mentioned previously that the model is pretrained on imagenet and image net has a specific mean and std that needs to be follow.

### Model

The model used is a pretrained ViT 224-16 patches. The pretrained model is transfer learned to classify between lung opacity, normal, viral pneumonia, COVID-19 chest x-ray images. The model was loaded using PyTorch.

```python
model = torch.hub.load('facebookresearch/deit:main', 'deit_base_patch16_224', pretrained=True)
### number of unique classes == 4
model.head = nn.Linear(768, 4)
```

Since this is a huge model, Tesla P100 GPU model is used from google colab.

### Training Phase

The training specs used are shown below.

```python
IMG_SIZE = 224
BATCH_SIZE = 32
LR = 2e-05
EPOCHS = 20
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.995)
device = "cuda"
```

Optimizer used is AdamW with learning rate of 2e-05 and an exponential learning rate scheduler is used. The criterion is cross entropy loss, since this is a multiclass classification problem and batch size used is 32.

Training loop uses a self made early stopping callback and save best model based on validation loss. The results are shown below.

| Epoch | Training loss | Validation loss | Training Accuracy | Validation Accuracy | 
|-------|---------------|-----------------|-------------------|---------------------|
|1      |0.443581       |0.243350         |0.8349             |0.9111               |
|2      |0.294692       |0.248474         |0.8942             |0.9059               |
|3      |0.245038       |0.198003         |0.9125             |0.9291               |
|4      |0.219254       |0.163702         |0.9210             |0.9390               |
|5      |0.204548       |0.147377         |0.9247             |0.9461               |
|6      |0.191227       |0.195680         |0.9301             |0.9338               |
|7      |0.181696       |0.153259         |0.9316             |0.9385               |
|8      |0.170775       |0.179045         |0.9389             |0.9395               |
|9      |0.167065       |0.152284         |0.9384             |0.9414               |
|10     |0.159125       |0.159929         |0.9428             |0.9414               |
|11     |0.153364       |0.142869         |0.9429             |0.9461               |
|12     |0.148766       |0.146074         |0.9464             |0.9485               |
|13     |0.141225       |0.164212         |0.9497             |0.9409               |
|14     |0.140871       |0.152537         |0.9477             |0.9447               |
|15     |0.131482       |0.155155         |0.9529             |0.9499               |
|16     |0.131830       |0.152081         |0.9527             |0.9466               |
|17     |0.122729       |0.149717         |0.9544             |0.9532               |
|18     |0.120725       |0.162608         |0.9563             |0.9442               |
|19     |0.114899       |0.178846         |0.9568             |0.9447               |
|20     |0.115127       |0.171766         |0.9579             |0.9414               |

The model that with the lowest validation loss is at epoch 11. Hence, the model at epoch 11, was saved. To import the model use `torch.load` function as shown below.

```python
import torch
model = torch.load(PATH)
```
Click [here](https://drive.google.com/file/d/1-0Lxrp20Ls1aEVclor2l0GeTO3XCKSQv/view?usp=sharing) to download model.

### Result Analysis

Using the test dataset with 2117 images, the model achieve a test accuracy of 95.8%. The confusion matrix is as shown below.

<img width=50% src="https://user-images.githubusercontent.com/67994195/123545864-a13e8b00-d79d-11eb-921d-bfb0d9b34c03.png">

### Conclusion

Although I haven't tried using any of the ResNet models, ViT gives a pretty decent result compared to other models in kaggle such as ResNet and efficient-net.

Just in case you guys wanted to read more on Vision Transformer, click [here](https://arxiv.org/abs/2010.11929) to read the paper.

