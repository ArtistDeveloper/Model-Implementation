import os
import math
import numpy as np
import requests
from PIL import Image
from inspect import isfunction
from functools import partial
from einops import rearrange, reduce
from einops.layers.torch import Rearrange
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.image as img
import matplotlib.pyplot as plt

from tqdm.auto import tqdm

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.utils as torch_utils
from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage, CenterCrop, Resize
from torch import nn, einsum
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision.utils import save_image

from Model_Implementation.GenerativeModel.dataset_class.rsna_breast_cancer import RSNADataset
from Model_Implementation.GenerativeModel.dataset_class.duke_dataset import DukeDataset

import ml_util


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class WeightStandardizedConv2d(nn.Conv2d):
    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        
        # 합성곱 연산을 수행하기 전에 필터 가중치를 정규화해야한다.
        weight = self.weight
        mean = reduce(weight, "o ... -> o 1 1 1", "mean")
        var = reduce(weight, "o ... -> o 1 1 1", partial(torch.var, unbiased=False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt() # 2 z = x−μ / σ + 0.0001(수치안정)
        
        return F.conv2d(
            x,
            normalized_weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = WeightStandardizedConv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x

        
class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2)) \
            if exists(time_emb_dim) \
            else None
        )

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()
        
        
    def forward(self, x, time_emb=None):
        scale_shift = None
        
        # TODO: Scale Shift에 대한 제대로 된 이해 필요
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, "b c -> b c 1 1") # Conv2D 수행하는 곳에 채널수와 크기를 맞춰주기 위함.
            scale_shift = time_emb.chunk(2, dim=1)
        
        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)
            


class Unet(nn.Module):
    def __init__(
        self,
        dim,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=1,
        self_condition=False,
        resnet_block_groups=4
    ):
        super().__init__()

        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels * (2 if self_condition else 1) # 1 * 1 = input_channel = 1
        
        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, init_dim, 1, padding=0)
        
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        
        time_dim = dim * 4
        print("time_dim: ", time_dim)
        
        # NOTE: partial로 나누는 것이 오히려 가독성이 안좋아서, 바로 ResnetBlcok으로 사용할 것이다.
        # block_klass = partial(ResnetBlock, groups=resnet_block_groups)
        # print("block_klass type: ", type(block_klass))
        
        # TODO: time_mlp 이해 및 구현 필요
        self.time_mlp = nn.Sequential(
            # SinusoidalPositionEmbeddings(dim),
            # nn.Linear(dim, time_dim),
            # nn.GELU(),
            # nn.Linear(time_dim, time_dim),
        )
        
    
    def forward(self, x, time, x_self_cond=None):
        # if self.self_condition:
        #     x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
        #     x = torch.cat((x_self_cond, x), dim=1)
        
        x = self.init_conv(x)
        r = x.clone()
        t = self.time_mlp(time)
        h = list()
        
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)
            
            x = block2(x, t)
            x = attn(x)
            h.append(x)
            
            x = downsample(x)
        
        # TODO: mid_block 작성
        
        
        # TODO: up_block 작성
            


def main():
    unet = Unet()
    


if __name__ == '__main__':
    main()