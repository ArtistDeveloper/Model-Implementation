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
class ForwardProcess():
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
    

class Diffusion(nn.Module):
    def __init__(
            self, 
            timesteps = 300
    ):
        # define beta schedule
        betas = ForwardProcess.linear_beta_schedule(timesteps=300)

        # define alphas 
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        def extract(a, t, x_shape): # TODO: self 붙여야 하나?
            batch_size = t.shape[0]
            out = a.gather(-1, t.cpu())
            return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)
        
        url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
        image = Image.open(requests.get(url, stream=True).raw) # PIL image of shape HWC
        plt.imshow(image)
        plt.show()


        image_size = 128
        transform = Compose([
            Resize(image_size),
            CenterCrop(image_size),
            ToTensor(), # turn into torch Tensor of shape CHW, divide by 255
            Lambda(lambda t: (t * 2) - 1),
            
        ])

        x_start = transform(image).unsqueeze(0)
        x_start.shape

        reverse_transform = Compose([
            Lambda(lambda t: (t + 1) / 2),
            Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
            Lambda(lambda t: t * 255.),
            Lambda(lambda t: t.numpy().astype(np.uint8)),
            ToPILImage(),
        ])

        test = reverse_transform(x_start.squeeze())
        plt.imshow(test)
        plt.show()


        # forward diffusion (using the nice property)
        def q_sample(x_start, t, noise=None):
            if noise is None:
                noise = torch.randn_like(x_start)

            sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
            sqrt_one_minus_alphas_cumprod_t = extract(
                sqrt_one_minus_alphas_cumprod, t, x_start.shape
            )

            return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
        

        def get_noisy_image(x_start, t):
            # add noise
            x_noisy = q_sample(x_start, t=t)

            # turn back into PIL image
            noisy_image = reverse_transform(x_noisy.squeeze())

            return noisy_image
        
        # take time step
        t = torch.tensor([40])

        noisy_image = get_noisy_image(x_start, t)
        print(type(noisy_image))
        plt.imshow(noisy_image)
        plt.show()



# 정리된 main
if __name__ == '__main__':
    model = Diffusion(timesteps=300)



# 다 때려 박아 놓은 것
# if __name__ == '__main__':
#     # forward process
#     timesteps = 300

#     # define beta schedule
#     betas = ForwardProcess.linear_beta_schedule(timesteps=timesteps)

#     # define alphas 
#     alphas = 1. - betas
#     alphas_cumprod = torch.cumprod(alphas, axis=0)
#     alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
#     sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

#     # calculations for diffusion q(x_t | x_{t-1}) and others
#     sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
#     sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

#     # calculations for posterior q(x_{t-1} | x_t, x_0)
#     posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

#     def extract(a, t, x_shape):
#         batch_size = t.shape[0]
#         out = a.gather(-1, t.cpu())
#         return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


#     url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
#     image = Image.open(requests.get(url, stream=True).raw) # PIL image of shape HWC
#     plt.imshow(image)
#     plt.show()


#     image_size = 128
#     transform = Compose([
#         Resize(image_size),
#         CenterCrop(image_size),
#         ToTensor(), # turn into torch Tensor of shape CHW, divide by 255
#         Lambda(lambda t: (t * 2) - 1),
        
#     ])

#     x_start = transform(image).unsqueeze(0)
#     x_start.shape

#     reverse_transform = Compose([
#         Lambda(lambda t: (t + 1) / 2),
#         Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
#         Lambda(lambda t: t * 255.),
#         Lambda(lambda t: t.numpy().astype(np.uint8)),
#         ToPILImage(),
#     ])

#     test = reverse_transform(x_start.squeeze())
#     plt.imshow(test)
#     plt.show()


#     # forward diffusion (using the nice property)
#     def q_sample(x_start, t, noise=None):
#         if noise is None:
#             noise = torch.randn_like(x_start)

#         sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
#         sqrt_one_minus_alphas_cumprod_t = extract(
#             sqrt_one_minus_alphas_cumprod, t, x_start.shape
#         )

#         return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    

#     def get_noisy_image(x_start, t):
#         # add noise
#         x_noisy = q_sample(x_start, t=t)

#         # turn back into PIL image
#         noisy_image = reverse_transform(x_noisy.squeeze())

#         return noisy_image
    
#     # take time step
#     t = torch.tensor([40])

#     noisy_image = get_noisy_image(x_start, t)
#     print(type(noisy_image))
#     plt.imshow(noisy_image)
#     plt.show()