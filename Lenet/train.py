from mnist_model import Lenet5
from torchsummary import summary

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


if __name__ == '__main__':
    device = set_device()
    model = create_model()
    loss_func = nn.CrossEntropyLoss()
    



