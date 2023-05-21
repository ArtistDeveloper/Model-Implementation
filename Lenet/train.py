from model import Lenet5

# torch library
from torchsummary import summary
from torchvision import transforms
from torchvision import datasets
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import utils
from torch.utils.data import DataLoader
import torch
import torch.nn as nn

import copy
import os
import urllib.request
import matplotlib.pyplot as plt
import numpy as np

 # GPU Setting
def set_device():
    if torch.cuda.is_available():
        device = torch.device('cuda');
    else:
        device = torch.device('cpu')

    print('Using pytorch version:', torch.__version__, 'Device:', device)
    return device


def create_model(device):
    model = Lenet5()    
    print(model)

    model.to(device)
    print(next(model.parameters()).device)

    summary(model, input_size=(1, 32, 32)) # image input size(mnist)
    return model


# 현재 lr을 계산하는 함수
def get_lr(opt):
    for param_group in opt.param_groups:
        return param_group['lr']
    

def show(train_data: datasets.MNIST, val_data: datasets.MNIST):
    x_train, y_train = train_data.data, train_data.targets
    x_val, y_val = val_data.data, train_data.targets

    ## 차원을 추가하여 B*C*H*W 가 되도록 만듬
    if len(x_train.shape) == 3:
        x_train = x_train.unsqueeze(dim=1)

    if len(x_val.shape) == 3:
        x_val = x_val.unsqueeze(dim=1)

    # images grid를 생성하고 출력
    # 총 40개 이미지, 행당 8개 이미지를 출력
    # x_train: (B x C x H x W) (60000, 1, 28, 28)
    # x_train[:40]: (B x C x H x W) (40, 1, 28, 28)
    x_grid = utils.make_grid(x_train[:40], nrow=8, padding=2)

    npimg: np = x_grid.numpy() # tensor를 numpy array로 변경합니다.
    npimg_tr = npimg.transpose((1,2,0)) # C*H*W(0, 1, 2)를 H*W*C(1, 2, 0)로 변경합니다.
    
    plt.imshow(npimg_tr, interpolation='nearest')
    plt.show()

"""
---------------------helper function start---------------------

"""



"""
---------------------helper function end---------------------

"""

if __name__ == '__main__':
    data_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])

    # download datasets
    data_path = './datasets'
    train_data = datasets.MNIST(data_path, train=True, download=True, transform=data_transform) # train 매개변수는 학습용 또는 테스트용 데이터셋 여부를 지정
    val_data = datasets.MNIST(data_path, train=False, download=True, transform=data_transform) 

    # sample image visualization
    show(train_data, val_data)

    # data loader
    train_dl = DataLoader(train_data, batch_size=32, shuffle=True)
    val_dl = DataLoader(val_data, batch_size=32, shuffle=False)

    