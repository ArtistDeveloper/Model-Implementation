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

    ## 차원을 추가하여 B*C*H*W 가 되도록 shape 변경
    if len(x_train.shape) == 3:
        x_train = x_train.unsqueeze(dim=1)

    if len(x_val.shape) == 3:
        x_val = x_val.unsqueeze(dim=1)

    """
    images grid를 생성하고 출력
    총 40개 이미지, 행당 8개 이미지를 출력
    x_train: (B x C x H x W) (60000, 1, 28, 28)
    x_train[:40]: (B x C x H x W) (40, 1, 28, 28)
    """
    x_grid = utils.make_grid(x_train[:40], nrow=8, padding=2)

    npimg: np = x_grid.numpy() # tensor를 numpy array로 변경
    npimg_tr = npimg.transpose((1,2,0)) # C*H*W(0, 1, 2)를 H*W*C(1, 2, 0)로 변경
    
    plt.imshow(npimg_tr, interpolation='nearest')
    plt.show()


# 배치당 performance metric을 계산하는 함수 정의
def metrics_batch(output, target):
    pred = output.argmax(dim=1, keepdim=True)
    corrects = pred.eq(target.view_as(pred)).sum().item()
    return corrects


# 배치당 loss를 계산하는 함수를 정의
def loss_batch(loss_func, output, target, opt=None):
    loss = loss_func(output, target)
    metric_b = metrics_batch(output, target)
    if opt is not None:
        opt.zero_grad()
        loss.backward()
        opt.step()
    return loss.item(), metric_b


# epoch당 loss와 performance metric을 계산하는 함수 정의
def loss_epoch(model: Lenet5, loss_func, dataset_loader, sanity_check=False, opt=None):
    running_loss = 0.0
    running_metric = 0.0
    len_data = len(dataset_loader.dataset)

    for xb, yb in dataset_loader:
        xb = xb.type(torch.float).to(device) # x데이터 배치
        yb = yb.to(device) # y데이터 배치
        output = model(xb) # 입력에 대한 예측 수행
        loss_bat, metric_bat = loss_batch(loss_func, output, yb, opt)
        running_loss += loss_bat

        if metric_bat is not None: 
            running_metric += metric_bat

        if sanity_check is True: # sanity_check가 True이면 1epoch만 학습합니다.
            break

    loss = running_loss / float(len_data)
    metric = running_metric / float(len_data)
    return loss, metric


# train_val 함수 정의
def train_val(model: Lenet5, params):
    num_epochs = params['num_epochs']
    loss_func = params['loss_func']
    opt = params['optimizer']
    train_loader = params['train_loader']
    val_loader = params['val_loader']
    sanity_check = params['sanity_check']
    lr_scheduler = params['lr_scheduler']
    path2weights = params['path2weights']

    loss_history = {
        'train': [],
        'val': [],
    }

    metric_history = {
        'train': [],
        'val': [],
    }

    # best model parameter를 저장합니다.
    # state_dict()를 통해 weight, bias등을 dict
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')

    for epoch in range(num_epochs):
        model.train() # train mode로 설정

        current_lr = get_lr(opt)
        print('Epoch {}/{}, current lr={}'.format(epoch, num_epochs-1, current_lr))
        train_loss, train_metric = loss_epoch(model, loss_func, train_loader, sanity_check, opt)

        loss_history['train'].append(train_loss)
        metric_history['train'].append(train_metric)

        model. eval() # eval mode로 설정
        with torch.no_grad(): # autograd engine을 꺼버림. 즉, 자동으로 gradient를 트래킹하지 않음
            val_loss, val_metric = loss_epoch(model, loss_func, val_loader, sanity_check)
            loss_history['val'].append(val_loss)
            metric_history['val'].append(val_metric)

        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), path2weights)
            print('Copied best model weights')

        lr_scheduler.step()

        print('train loss: %.6f, dev loss: %.6f, accuracy: %.2f' %(train_loss, val_loss, 100*val_metric))
        print('-' * 10)

    model.load_state_dict(best_model_wts)
    return model, loss_history, metric_history



if __name__ == '__main__':
    lr = 0.001

    # cuda 세팅
    device = set_device()

    data_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])

    # 데이터셋 다운로드
    DATA_PATH = './datasets'
    train_data = datasets.MNIST(DATA_PATH, train=True, download=True, transform=data_transform) # train 매개변수는 학습용 또는 테스트용 데이터셋 여부를 지정
    val_data = datasets.MNIST(DATA_PATH, train=False, download=True, transform=data_transform) 

    # 이미지 시각화 테스트
    # show(train_data, val_data)

    # 데이터 로더 객체 생성
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

    # CUDA로 모델 이동
    model = Lenet5()
    model.to(device)
    # print(model)
    # print(next(model.parameters()).device)
    
    # 모델 summary 출력
    summary(model, input_size=(1, 32, 32))

    # loss 함수 정의 및 optimizer 설정
    loss_func = nn.CrossEntropyLoss(reduction='sum')
    opt = optim.Adam(model.parameters(), lr=lr)
    lr_scheduler = CosineAnnealingLR(opt, T_max=2, eta_min=1e-05)


    # 학습된 모델의 가중치를 저장할 폴더를 만듭니다.
    os.makedirs('./models', exist_ok=True)

    # 하이퍼파라미터 설정
    params_train={
    "num_epochs": 3,
    "optimizer": opt,
    "loss_func": loss_func,
    "train_loader": train_loader,
    "val_loader": val_loader,
    "sanity_check": False,
    "lr_scheduler": lr_scheduler,
    "path2weights": "./models/LeNet-5.pt",
    }
    
    # 모델 및 옵티마이저의 state_dict() 확인
    print("Model's state_dict")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    print("Optimizer's state_dict")
    for var_name in opt.state_dict():
        print(var_name, "\t", opt.state_dict()[var_name])

    # 모델 학습
    model, loss_hist, metric_hist = train_val(model, params_train)
    