import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
import os


class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential() # identity인 경우. 비어있는 컨테이너이며, forward에서 x를 입력받아 residual 연산을 할 수 있도록 만든다.
        if stride != 1: # stride가 1이 아니라면, identity mapping이 아닌 경우이다. 이 경우에는 projection을 수행하여 forward에서 계산될 수 있도록 만든다.
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )
             
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out)) # 여기까지가 F(x)
        out += self.shortcut(x) # F(x) + x
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=2):
        super(ResNet, self).__init__()
        self.in_planes = 64

        # 입력 이미지의 사이즈: 128x128
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False) # shape= 1x128x128 -> shape= 64x128x128
        self.bn = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1) # shape= 64x128x128 -> shape= 64x128x128
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2) # shape= 64x128x128 -> shape= 128x64x64
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2) # shape= 128x64x64 -> shape= 256x32x32
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2) # shape= 256x32x32 -> shape= 512x16x16
        self.linear = nn.Linear(512, num_classes)


    def _make_layer(self, block, out_planes, num_block, stride):
        strides = [stride] + ([1] * (num_block - 1))

        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, out_planes, stride))
            self.in_planes = out_planes

        return nn.Sequential(*layers)    


    def forward(self, x):
        out = self.bn(self.conv1(x))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 16) # input tensor=512x16x16, kernel_size=16 -> 512x1x1
        out = out.view(out.size(0), -1) # 512x1x1 -> 512x1
        out = self.linear(out)        
        return out


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])