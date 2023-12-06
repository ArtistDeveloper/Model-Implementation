import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from random import sample

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class AutoEncoderConv(nn.Module):
    def __init__(self):
        super(AutoEncoderConv, self).__init__()
        
        self.encoder = nn.Sequential(
            # 28x28 -> 14x14
            nn.Conv2d(1, 64, 3, stride=1, padding=1),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            
            # 14x14 -> 7x7
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.ReLU(),
            
            # 7x7 -> 3x3
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.Conv2d(256, 512, 4, stride=2, padding=1),
            nn.ReLU(),
        )
        
        self.decoder = nn.Sequential(
            # nn.ConvTranspose2d(512, 256, 3, )
            nn.ConvTranspose2d(512, 256, 3, stride=2, padding=0, output_padding=0)
        )
        
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded


def main():
    return NotImplemented


if __name__ == '__main__':
    main()