from model_for_MNIST import Lenet5
from torchsummary import summary
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR

import torch
import torch.nn as nn

 # GPU Setting
def set_device():
    if torch.cuda.is_available():
        device = torch.device('cuda');
    else:
        device = torch.device('cpu')

    print('Using pytorch version:', torch.__version__, 'Device:', device)
    return device

def create_model():
    model = Lenet5()
    print(model)

    model.to(device)
    print(next(model.parameters()).device)

    summary(model, input_size=(1, 32, 32))
    return model

def get_lr(opt):
    for param_group in opt.param_groups:
        return param_group['lr']


"""
---------------------helper function start---------------------

"""

# 배치당 performance metric을 계산하는 함수 정의
def metrics_batch(output, target):
    pred = output.argmax(dim=1, keepdim=True)
    corrects = pred.eq(target.view_as(pred)).sum().item()
    return corrects

# epoch당 loss와 performance metric을 계산하는 함수 정의
def loss_batch(loss_func, output, target, opt=None):
    loss = loss_func(output, target)


"""
---------------------helper function end---------------------

"""

if __name__ == '__main__':
    device = set_device()
    model = create_model()
    loss_func = nn.CrossEntropyLoss()
    opt = optim.SGD(model.parameters(), lr=0.001)
    lr_scheduler = CosineAnnealingLR(opt, T_max=2, eta_min=1e-05)



