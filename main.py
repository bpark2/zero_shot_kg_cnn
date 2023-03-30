import gc
import os
import matplotlib.pyplot as plt
import numpy as np
import PIL
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import models
import torchvision.transforms as T
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

training_data = datasets.ImageNet(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.ImageNet(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

model = models.resnet101(weights=models.ResNet50_Weights.IMAGENET1K_V1)

def evaluate(mode, device, eval_loader):
    model.eval()
    loss = 0
    corr = 0
    with torch.no_grad():
        for data, target in eval_loader:
            data, target = data.to('cuda'), target.to('cuda')
            output = model(data)
            pred = output.argmax(1,keepdim=True)
            print(pred, target)
            corr += pred.eq(target.view_as(pred)).sum().item()

    loss /= len(eval_loader.dataset)

    print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(
        corr, len(eval_loader.dataset),
        100. * corr / len(eval_loader.dataset)))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Cuda Available: ",torch.cuda.is_available())

epoch_num = 15
batch_size = 64
learning_rate = 0.001

evaluate(model,device,test_data)