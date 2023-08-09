import math
import numpy as np
from inspect import isfunction
from functools import partial

import matplotlib.pyplot as plt
from tqdm.auto import tqdm
# from einops import rearrange, reduce
# from einops.layers.torch import Rearrange

from PIL import Image
import requests

import torch
from torch import nn, einsum
import torch.nn.functional as F

import matplotlib.image as img
import matplotlib.pyplot as plt

from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage, CenterCrop, Resize


# Network helper


# Defining the forward diffusion process
class ForwardBetaSchedule():
    @staticmethod
    def cosine_beta_schedule(timesteps, s=0.008):
        """
        cosine schedule as proposed in https://arxiv.org/abs/2102.09672
        - cosine 스케줄링은 저해상도에서 유용. 고해상도 이미지에서는 linear로 충분
        """
        
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
    
    @staticmethod
    def linear_beta_schedule(timesteps):
        beta_start = 0.0001
        beta_end = 0.02
        return torch.linspace(beta_start, beta_end, timesteps)
    
    @staticmethod
    def quadratic_beta_schedule(timesteps):
        beta_start = 0.0001
        beta_end = 0.02
        return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2
    
    @staticmethod
    def sigmoid_beta_schedule(timesteps):
        beta_start = 0.0001
        beta_end = 0.02
        betas = torch.linspace(-6, 6, timesteps)
        return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start



def load_image():
    url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    image = Image.open(requests.get(url, stream=True).raw) # PIL image of shape HWC
    plt.imshow(image)
    plt.show()
    return image


def extract(alphas, t, x_shape):
    batch_size = t.shape[0]
    out = alphas.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


class Diffusion(nn.Module):
    def __init__(
            self, 
            timesteps = 300
    ):
        # beta schedule 정의
        self.betas = ForwardBetaSchedule.linear_beta_schedule(timesteps=timesteps)

        # alphas 정의 
        self.alphas = 1. - self.betas # alpha: 1 - beta
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0) # alpha bar: start ~ timestep까지의 누적곱
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0) # 마지막 숫자 제외한 뒤 맨 앞(왼쪽)에 값 1.0으로 한 개의 패딩 추가
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        # (ddpm eq.4) diffusion q(x_t | x_{t-1}) 및 다른 것들 계산
        # 이 방식으로 샘플링을 할 때, 한 번에 노이즈를 다 입힌 이미지로 변환가능하다. 
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod) # 정규분포의 평균에서 사용할 값
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod) # 정규분포의 표준편차에서 사용할 값

        # 사후 확률 q(x_{t-1} | x_t, x_0)에 대한 계산
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

    
    # 포워드 디퓨전 (using the nice property)
    def q_sample(self, x_start:torch.Tensor, t:torch.Tensor, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    

    def get_noisy_image(self, x_start, t):
        x_noisy = self.q_sample(x_start, t=t)
        
        # 역변환을 정의하여, [-1, 1]의 값을 포함하는 PyTorch 텐서를 받아 이를 다시 PIL 이미지로 변환
        reverse_transform = Compose([
            Lambda(lambda t: (t + 1) / 2),
            Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
            Lambda(lambda t: t * 255.),
            Lambda(lambda t: t.numpy().astype(np.uint8)),
            ToPILImage(), # tensor or ndarray to PIL image
        ])

        noisy_image = reverse_transform(x_noisy.squeeze())

        return noisy_image

    
    def test_forward_process(self):
        image = load_image()
        
        image_size = 128
        transform = Compose([
            Resize(image_size),
            CenterCrop(image_size),
            ToTensor(), # C,H,W shape의 tensor로 변환하며 255로 나누어 0~1 사이의 값으로 정규화
            Lambda(lambda t: (t * 2) - 1), # [-1, 1] 사이로 값 정규화
        ])

        # preprocessing image, it is start of x
        x_start = transform(image).unsqueeze(0) # add the '1' dimension for 0 dim index. [3,128,128] -> [1,3,128,128]

        # take time step
        t = torch.tensor([40])

        noisy_image = self.get_noisy_image(x_start, t)
        print(type(noisy_image))
        plt.imshow(noisy_image)
        plt.show()

        


# 정리된 main
if __name__ == '__main__':
    model = Diffusion(timesteps=300)
    model.test_forward_process()

