import torch
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import utils
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import matplotlib.pyplot as plt
import time


def show(img, path):
    img = img.numpy() # Tensor -> numpy array
    img = img.transpose([1,2,0]) # C x H x W -> H x W x C
    plt.imshow(img, interpolation='nearest')
    plt.savefig(path + "/test.png")


def main():
    device = torch.device('cuda:5' if torch.cuda.is_available() else 'cpu')
    img_save_path = r"/workspace/Model_Implementation/GenerativeModel/gan/results"
    
    # Set Data path
    datapath = './data'
    os.makedirs(datapath, exist_ok=True)

    # Pre-process
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    # Laod MNIST
    train_dataset = datasets.MNIST(datapath, train=True, download=True, transform=trans)
    
    img, label = train_dataset.data, train_dataset.targets
    print('img.shape:', img.shape)
    print('label.shape:', label.shape)

    # Make it to 4D Tensor1
    # 기존 : (#Batch) x (height) x (width) -> (#Batch) x (#channel) x (height) x(width)
    if len(img.shape) == 3:
        img = img.unsqueeze(1)
    print('Unsqueezed img.shape:', img.shape)

    # Visualize
    img_grid = utils.make_grid(img[:40], ncol=8, padding=2)
    show(img_grid, img_save_path)


if __name__=='__main__':
    main()