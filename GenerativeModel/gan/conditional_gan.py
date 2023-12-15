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


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_classes = 10
        self.nz = 100
        self.input_size = (1, 28, 28)
        
        # Noise와 label을 결합하는 용도인 label embedding matrix를 생성
        self.label_emb = nn.Embedding(self.num_classes, self.num_classes)
        print("label_emb.shape: ", self.label_emb.shape)
        
        # Generator
        self.gen = nn.Sequential(
            nn.Linear(self.nz + self.num_classes, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256,512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512,1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024,int(np.prod(self.input_size))),
            nn.Tanh()
        )
    
    def forward(self, noise, labels):
        # noise와 label의 결합
        gen_input = torch.cat((self.label_emb(labels), noise), -1)
        x = self.gen(gen_input)
        x = x.view(x.size(0), *self.input_size)
        return x


def check_data(train_dataset, img_save_path):
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
    print("img_grid type: ", type(img_grid))
    print("img_grid shape: ", img_grid.shape)
    
    show(img_grid, img_save_path)


def show(img, path):
    img = img.numpy() # Tensor -> numpy array
    img = img.transpose([1,2,0]) # C x H x W -> H x W x C
    plt.imshow(img, interpolation='nearest')
    plt.savefig(path + "/test.png")


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
    
    # Check data
    check_data(train_dataset, img_save_path)
    
    # DatLoader
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    print(len(train_loader))


if __name__=='__main__':
    main()