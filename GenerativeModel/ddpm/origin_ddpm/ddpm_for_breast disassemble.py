import os
import math
from pathlib import Path
from inspect import isfunction
from functools import partial

import numpy as np
from einops import rearrange, reduce
from einops.layers.torch import Rearrange
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.utils as torch_utils
from torch import nn, einsum

from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage, CenterCrop, Resize
from torchvision import transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter


from Model_Implementation.GenerativeModel.dataset_class.rsna_breast_cancer import RSNADataset
from Model_Implementation.GenerativeModel.dataset_class.duke_dataset import DukeDataset

import ml_util


class GlobalVar:
    # Select cuda device
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device_ids = [0, 1]
    
    # Tensorboard
    writer = SummaryWriter()
    global_step = 0
    
    # TODO: cfg 변수 여기로 옮겨, main(), train() 함수에서 사용할 수 있도록 변경 -> 메소드의 파라미터 줄일 수 있음.
    


def exists(x):
    return x is not None


def default(val, d):
    """
    기본 값이 존재한다면 val을 return하고, 기본값 자체가 계산이 필요한 경우
    함수를 통해 계산된 결과를 제공할 수 있다.
    """
    if exists(val):
        return val
    return d() if isfunction(d) else d


def num_to_groups(num, divisor):
    groups = num // divisor # 4 // 128 = 0
    remainder = num % divisor # 4 % 128 = 4
    arr = [divisor] * groups # divisor값을 갖는 요소를 groups의 횟수만큼 반복해서 list에 넣음. divisor: 3, groups: 4라면 [3, 3, 3, 3]이다.
    if remainder > 0:
        arr.append(remainder)
    return arr



class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn # fn: 함수 전달 받음

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x 


def Upsample(dim, dim_out=None):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding=1),
    )


def Downsample(dim, dim_out=None):
    # No More Strided Convolutions or Pooling
    return nn.Sequential(
        Rearrange("b c (h p1) (w p2) -> b (c p1 p2) h w", p1=2, p2=2),
        nn.Conv2d(dim * 4, default(dim_out, dim), 1),
    )



class SinusoidalPositionEmbeddings(nn.Module):
    """Positional embedding"""
    
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1) # math.log(10000) = ln 9.2
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings) # 1.0 ~ 0.0 사이의 값
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings



class WeightStandardizedConv2d(nn.Conv2d):
    """
    https://arxiv.org/abs/1903.10520
    weight standardization purportedly works synergistically with group normalization
    """

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        
        # 합성곱 연산을 수행하기 전에 필터 가중치를 정규화해야한다.
        weight = self.weight
        mean = reduce(weight, "o ... -> o 1 1 1", "mean")
        var = reduce(weight, "o ... -> o 1 1 1", partial(torch.var, unbiased=False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt() #  z = x−μ / σ + 0.0001(수치안정)
        
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
    """https://arxiv.org/abs/1512.03385"""

    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2)) \
            if exists(time_emb_dim) \
            else None
        )

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity() # 채널크기가 같으면 그대로, 아니면 conv2d로 채널 수 맞춰주기

    def forward(self, x, time_emb=None):
        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, "b c -> b c 1 1") # Conv2D 수행하는 곳에 채널수와 크기를 맞춰주기 위함.
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)



"""Attention Module"""
class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )
        q = q * self.scale

        sim = einsum("b h d i, b h d j -> b h i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = einsum("b h i j, b h d j -> b h i d", attn, v)
        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
        return self.to_out(out)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1), 
                                    nn.GroupNorm(1, dim))

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        return self.to_out(out)



"""Group normalization"""
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)



class Unet(nn.Module):
    """
    epsilon(x_t, t) 네트워크의 역할은 노이즈 이미지와 각각의 노이즈 레벨을 batch로 받아서 입력에 추가된 노이즈를 output으로 출력한다:
    (batch_size, num_channels, height, width) 형태의 노이즈 이미지와 (batch_size, 1) 형태의 노이즈 레벨 배치를 입력으로 받아서,
    (batch_size, num_channels, height, width) 형태의 텐서를 반환한다.

    네트워크의 구성:
    먼저 노이즈 이미지에 컨볼루션 레이어가 적용되고 position embedding이 노이즈 레벨에 대해 계산된다.

    다음으로 일련의 다운 샘플링 단계가 적용된다. 다운샘플링 단계는 
    2개의 ResNet 블록 + groupnorm + attention + residual connection + 다운 샘플 연산으로 구성된다.

    네트워크 중간에서 다시 ResNet block이 적용되고, attention으로 interleaved된다.

    다음으로 일련의 업샘플링 단계가 적용된다. 각 업샘플링 단계는 
    2개의 ResNet 블록 + groupnorm + attention + residual connection + 업샘플링 연산으로 구성된다.

    마지막으로 ResNet 블록과 컨볼루션 레이어가 적용된다.
    """
    
    def __init__(
        self,
        dim, # image_size
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=1, # image_channel
        # self_condition=False,
        resnet_block_groups=4,
    ):
        super().__init__()

        # 차원 결정
        self.channels = 1
        self.self_condition = False
        input_channels = 1

        init_dim = 256
        
        self.init_conv = nn.Conv2d(1, 256, kernel_size=1, padding=0)

        dims = [256, 256, 512, 1024] # [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = [(256, 256), (256, 512), (512, 1024)] # list(zip(dims[:-1], dims[1:]))

        
        block_klass = partial(ResnetBlock, groups=resnet_block_groups)


        time_dim = 1024 # dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(256),
            nn.Linear(256, 1024),
            nn.GELU(),
            nn.Linear(1024, 1024),
        )

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = 3 # len(in_out)
        
        
        #region down_layer for
        # for idx, (dim_in, dim_out) in enumerate(in_out):
        #     is_last = idx >= (num_resolutions - 1)

        #     self.downs.append(
        #         nn.ModuleList(
        #             [
        #                 block_klass(dim_in, dim_in, time_emb_dim=time_dim),
        #                 block_klass(dim_in, dim_in, time_emb_dim=time_dim),
        #                 Residual(PreNorm(dim_in, LinearAttention(dim_in))),
        #                 Downsample(dim_in, dim_out)
        #                 if not is_last
        #                 else nn.Conv2d(dim_in, dim_out, 3, padding=1),
        #             ]
        #         )
        #     )
        #endregion
        
        
        #region down_layer
        # first
        is_last = False
        self.downs.append(
            nn.ModuleList(
                [
                        block_klass(256, 256, time_emb_dim=1024),
                        block_klass(256, 256, time_emb_dim=1024),
                        Residual(PreNorm(256, LinearAttention(256))),
                        Downsample(256, 256)
                        if not is_last
                        else nn.Conv2d(256, 256, 3, padding=1),
                ]
            )
        )
        print('------------------------------------------')
        print("self.downs1: ", self.downs)
        
        # second
        is_last = False
        self.downs.append(
            nn.ModuleList(
                [
                        block_klass(256, 256, time_emb_dim=1024),
                        block_klass(256, 256, time_emb_dim=1024),
                        Residual(PreNorm(256, LinearAttention(256))),
                        Downsample(256, 512)
                        if not is_last
                        else nn.Conv2d(256, 512, 3, padding=1),
                ]
            )
        )
        print('------------------------------------------')
        print("self.downs2: ", self.downs)
        
        # third
        is_last = True
        self.downs.append(
            nn.ModuleList(
                [
                        block_klass(512, 512, time_emb_dim=1024),
                        block_klass(512, 512, time_emb_dim=1024),
                        Residual(PreNorm(512, LinearAttention(512))),
                        Downsample(512, 1024)
                        if not is_last
                        else nn.Conv2d(512, 1024, 3, padding=1),
                ]
            )
        )
        print('------------------------------------------')
        print("self.downs3: ", self.downs)
        
        # NOTE : what is Image size??
        #endregion         



        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                        block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                        Upsample(dim_out, dim_in)
                        if not is_last
                        else nn.Conv2d(dim_out, dim_in, 3, padding=1),
                    ]
                )
            )

        self.out_dim = default(out_dim, channels)

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim=time_dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)
        
    
    def forward(self, x, time, x_self_cond=None):
        """
        x: random noise image
        time: 0~Timestep 사이의 랜덤 int값    
        """
        count = 0
        
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim=1)

        print(f"Input x.shape: {x.shape} / iter: {count}")
        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            print(f"x.shape: {x.shape} / iter: {count}")
            
            h.append(x)
            print(f"h.len: {len(h)} / iter: {count}")

            x = block2(x, t)
            x = attn(x)
            h.append(x)

            x = downsample(x)
            print(f"x_downsample.shape: {x.shape} / iter: {count}")

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t)
            x = attn(x)

            x = upsample(x)

        x = torch.cat((x, r), dim=1)

        x = self.final_res_block(x, t)
        
        count = count + 1
        return self.final_conv(x)


# 포워드 디퓨전 프로세스 정의
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


def extract(value, target_t, x_shape):
    """
    처음부터 끝까지 구해놓은 value 텐서와 목표하는 timestep인 텐서 target_t를 받아
    target_t를 index로 씀으로 alphas에서 값을 하나 추출한다.
    out = value.gather(-1, target_t.cpu())에서 
    value shape가 300이면 300개 중 target(여기선 40)의 인덱스의 위치의 값을 가져온다.
    """
    batch_size = target_t.shape[0]
    out = value.gather(-1, target_t.cpu()) # dim=-1은 차원에서 마지막 차원을 뜻한다.(무조건 열이 되도록 해놓은 것일듯), index: value 텐서에서 접근할 기준 인덱스들

    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(target_t.device) # (?, ?, ?, ?)shpae으로 값 return 


# TODO: diffusion 클래스 안에 UNet모델 객체 삽입하기
class DiffusionUtils(nn.Module):
    t_timesteps = 300

    def __init__(
            self, 
            total_timesteps = 300
    ):
        """
        아래 변수 모두 timestep만큼의 size를 가지고 있다. 학습이랑 샘플링 때 필요한 t에 맞게 index를 가져가면 된다.

        # Forward process
        alphas: 
            ddpm (eq.4)위에 alpha의 정의가 있다. 1 - beta가 알파이다.

        alphas_cumprod: 
            ddpm (eq.4)위에 alpha_bar의 정의가 있다. 1~t까지의 누적곱을 뜻한다.

        alphas_cumprod_prev: 
            샘플링 과정(eq.6)에서 뮤 틸드를 구할 때 필요. 
            알파 바의 t-1이 필요하기에 맨 마지막을 제거하고 맨 처음에 알파가 하나도 없었다는 뜻의 1을 집어넣는다.

        # Training
        sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod:
                학습 과정에서 필요. DDPM의 Algorithm 1 Training 참고.

        # Sampling
        sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas):
            루트 알파 바의 역수가 샘플링 과정에서 필요하다. 
        
        # ??
        posterior_variance:
            사후 확률 q(x_{t-1} | x_t, x_0)(eq.6.) 계산에서 분산에 사용되는 값. beta tilde로 표현된다.
            KL divergence를 사용하여 바로 pθ(xt−1|xt)와 forward process의 posterior와 비교하는데, 
            이 때 X_0를 조건으로 하면 forward process의 posterior를 추적할 수 있다.
            즉 구하고자 하는 t와, 처음 시작지점 x_0를 조건으로 주면 x_{t-1}의 노이즈 상태를 얻을 수 있다.
        """
        # beta schedule 정의
        DiffusionUtils.t_timesteps = total_timesteps
        self.betas = ForwardBetaSchedule.linear_beta_schedule(timesteps=total_timesteps)

        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0) 

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod) 
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod) 

        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

    
    def q_sample(self, x_start:torch.Tensor, target_t:torch.Tensor, noise=None):
        """
        forward process이다. 목표하는 타입스텝까지의 tensor들을 extract해서 텐서를 가져온다.
        
        sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise의 이해
        식 4를 참조하면 평균에서 sqrt{hat{alpha}} * x_0 (여기선 x_start)를 확인할 수 있으며
        표준편차에서 (1 - hat{alpha_t}) * I 를 볼 수 있다. 해당 수식과 상당히 유사. (I가 noise로 추정. )
        즉, 평균에서 표준편차만큼 계속 더해주면 노이즈라는건가?

        기존 노이즈 * 시작 이미지 + 추가 노이즈
        """
        if noise is None:
            noise = torch.randn_like(x_start) # torch.randn(): 정규분포를 이용하여 생성

        # 학습할 때 아래 값들이 필요하다.
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, target_t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, target_t, x_start.shape)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    
    def p_losses(self, denoise_model, x_start, t, noise=None, loss_type="l1"):
        """
        x_noisy: 알고리즘 1-5에서 epsilon theta의 첫 번째 매개변수로 들어갈 값
        t: uniform 하게 뽑은 랜덤한 int값 (배치사이즈만큼의 개수를 가진다.)
        """
        
        if noise is None:
            noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, target_t=t, noise=noise)
        predicted_noise = denoise_model(x_noisy, t)

        if loss_type == 'l1':
            loss = F.l1_loss(noise, predicted_noise)
        elif loss_type == 'l2':
            loss = F.mse_loss(noise, predicted_noise)
        elif loss_type == "huber":
            loss = F.smooth_l1_loss(noise, predicted_noise)
        else:
            raise NotImplementedError()

        return loss
    
    """
    Sampling
    디퓨전 모델에서 새로운 이미지를 생성하려면 디퓨전 프로세스(포워드 프로세스)를 역전시켜 가우스 분포에서
    순수한 노이즈를 샘플링하는 T에서 시작한 다음, 신경망을 사용하여 학습한 조건부 확률을 사용하여 점진적으로 
    노이즈를 제거하여 timestep t=0에 도달한다. 노이즈 예측기(model)을 사용하여 평균의 reparameterization(재측정?)값을
    연결하면 노이즈가 약간 덜 제거된 이미지 x{t-1}을 도출할 수 있다. 분산은 미리 알고 있다는 것을 염두에 두어라.

    이상적으로 실제 데이터 분포에서 가져온 것처럼 보이는 이미지로 끝내는 것이 좋다.
    """


    @torch.no_grad()
    def p_sample(self, model, x_image, t, t_index):
        # t.shape = [64] (배치사이즈 크기)

        betas_t = extract(self.betas, t, x_image.shape) # t까지의 betas 얻어오기 shape = [batchsize, 1, 1, 1]
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_image.shape) # shape = [batchsize, 1, 1, 1]
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t, x_image.shape) # shape = [batchsize, 1, 1, 1]
        
        # 논문 Eq.11
        # 모델(노이즈 예측기)을 사용하여 평균을 예측한다.
        model_mean = sqrt_recip_alphas_t * (
            x_image - betas_t * model(x_image, t) / sqrt_one_minus_alphas_cumprod_t
        )

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = extract(self.posterior_variance, t, x_image.shape)
            noise = torch.randn_like(x_image)
            
            # 알고리즘 2, 4번째 줄
            return model_mean + torch.sqrt(posterior_variance_t) * noise 


    @torch.no_grad()
    def p_sample_loop(self, model, shape, skip_step):
        """
        알고리즘 2를 구현한 것이다. (모든 이미지 반환 포함)
        샘플링된 이미지들을 반환하기 위해 imgs 배열에 샘플링된 이미지를 추가한다.
        샘플링의 timestep은 마지막부터 시작하여 1까지 반복하여 list를 reverse하여 반복한다.
        """
        device = next(model.parameters()).device

        batch_size = shape[0] # shpae: batch, channel, img_size, img_size       
        
        # 이미지는 순수한 노이즈에서 시작 (배치의 각 예제에 대해)
        batch_imgs = torch.randn(shape, device=device) # shape=(batch_size, channels, img_size, img_size) [4,1,28,28]
        total_imgs = []
        for i in tqdm(reversed(range(0, DiffusionUtils.t_timesteps)), desc='sampling loop time step', total=DiffusionUtils.t_timesteps):
            batch_imgs = self.p_sample(model, batch_imgs, torch.full((batch_size,), i, device=device, dtype=torch.long), i)
            total_imgs.append(batch_imgs.cpu().numpy())

        # timestep의 이미지만큼 sampling해서 이미지들을 반환한다. timestep이 300이면 300개의 이미지가 담긴 1차원 벡터 반환
        print("len total_imgs: ", len(total_imgs))
        extracted_imgs = total_imgs[0 : DiffusionUtils.t_timesteps : 100]
        extracted_imgs.append(total_imgs[DiffusionUtils.t_timesteps - 1])
        
        return extracted_imgs

    
    @torch.no_grad()
    def sample(self, model, image_size, batch_size=16, channels=3, skip_step=1):
        return self.p_sample_loop(model, shape=(batch_size, channels, image_size, image_size), skip_step=skip_step)
    

    def get_noisy_image(self, x_start, target_t):
        x_noisy = self.q_sample(x_start, target_t)
        
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

    
    def test_forward_process(self, image):
        image_size = 128
        transform = Compose([
            Resize(image_size),
            CenterCrop(image_size),
            ToTensor(), # C,H,W shape의 tensor로 변환하며 255로 나누어 0~1 사이의 값으로 정규화
            Lambda(lambda t: (t * 2) - 1), # [-1, 1] 사이로 값 정규화
        ])

        # 이미지 전처리 후 '0'dim 인덱스에 '1'인 차원을 추가한다. ex) [3,128,128] -> [1,3,128,128]        
        # 원본 이미지
        x_start = transform(image).unsqueeze(0) 

        # 특정 타임 스텝 취해보기 (여기선 40)
        t = torch.tensor([40])

        # 노이즈 이미지 얻기
        noisy_image = self.get_noisy_image(x_start, t)
        
        print(type(noisy_image))
        plt.imshow(noisy_image)
        plt.savefig('./noisy.png')


def get_rsna_dataloader(png_dir, train_batchsize=32, image_size=256):
    img_transform = transforms.Compose(
        [
            transforms.Resize(size=(image_size, image_size)),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: (t * 2) - 1),  # NOTE: Scale between [-1, 1]
        ]
    )
    
    dataset = RSNADataset(png_dir, img_transform)
    dataset_size = len(dataset)
    print("dataset_size: ", dataset_size)

    train_loader = DataLoader(dataset, batch_size=train_batchsize, shuffle=True)

    return train_loader


def get_duke_dataloader(png_dir, train_batchsize=32, img_size=256, num_workers=8):
    img_transform = transforms.Compose(
        [
            transforms.Resize(size=(img_size, img_size)),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: (t * 2) - 1),  # NOTE: Scale between [-1, 1]
        ]
    )
    
    dataset = DukeDataset(png_dir, img_transform)
    print(len(dataset))
    
    train_lodaer = DataLoader(dataset, batch_size=train_batchsize, shuffle=True, num_workers=num_workers)
    
    return train_lodaer
    

def sample_image(model, diffusion_model, epoch, batch_size, img_size, channels, results_folder):
    # milestone = step // save_and_sample_every
    batches = num_to_groups(4, batch_size) # 4, 8, batches = [4]

    # map: 리스트의 요소를 지정된 함수로 처리해준다. map(function, iterable)
    # 첫 번째 배치의 이미지 리스트를 가져오고, 각 이미지를 텐서로 변환. 이미지는 batches * timestep의 개수만큼 반환되어 온다.
    # 여기서 batches는 4이고, TIMESTEPS은 10이라면 총 40개의 이미지를 가져오게 된다.
    all_images_list = list(map(lambda n: diffusion_model.sample(model, img_size, batch_size=n, channels=channels, skip_step=100), batches))
    image_tensors = [torch.tensor(image) for image in all_images_list[0]] 
    all_images_tensor = torch.cat(image_tensors, dim=0) 
    all_images_tensor = (all_images_tensor + 1) * 0.5 # 이미지 값 범위를 [0, 1]로 조정
    print("all_images_tensor shape: ", all_images_tensor.shape)
    save_path = os.path.join(results_folder, f'sample-{epoch}.png')    
    save_image(all_images_tensor, save_path, nrow=4)
    


def train(cfg, optimizer, model, diffusion_model, dataloader, loss_list):
    MODEL_SAVE_STEP = 20
    
    cfg_params = cfg['params']
    cfg_paths = cfg['paths']
    
    img_size = cfg_params['img_size']
    channels = cfg_params['channels']
    timesteps = cfg_params['timesteps']
    epochs = cfg_params['epochs']
    max_grad_norm = cfg_params['max_grad_norm']
    
    model_save_path = cfg_paths['model_save_path']
    diffusion_results_path = cfg_paths['diffusion_results_path']
    
    
    for epoch in tqdm(range(epochs)):
        for step, image_batch in enumerate(tqdm(dataloader)):
            optimizer.zero_grad()        

            batch_size = image_batch.shape[0]
            image_batch = image_batch.to(GlobalVar.device)

            # 알고리즘 1, 3번째 줄: 배치의 모든 예제에 대해 균일하게 t를 샘플링한다.
            t = torch.randint(0, timesteps, (batch_size,), device=GlobalVar.device).long()            

            loss = diffusion_model.p_losses(model, image_batch, t, noise=None, loss_type="huber")
            GlobalVar.writer.add_scalar('Loss/train', loss.item(), GlobalVar.global_step)
            GlobalVar.global_step += 1

            if step % 100 == 0:
                print("Loss:", loss.item())
                loss_list.append(loss.item())

            loss.backward()
            torch_utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            
            
            if step != 0 and step % cfg_params['save_and_sample_every'] == 0:            
                sample_image(
                    model=model,
                    diffusion_model=diffusion_model,
                    epoch=epoch,
                    batch_size=batch_size,
                    img_size=img_size,
                    channels=channels,
                    results_folder=diffusion_results_path
                )

        
        # scheduler.step()
        
        if epoch % MODEL_SAVE_STEP == 0:
            torch.save(model, f"{model_save_path}/{epoch}_full_model.pth") 

        
        if epoch == (epochs - 1):
            torch.save(model, f"{model_save_path}/{epoch}_full_model.pth") 
    



def main():
    YAML_PATH = r"/workspace/Model_Implementation/GenerativeModel/ddpm/origin_ddpm/configs/256x256_diffusion.yaml"
    
    cfg = ml_util.load_config(YAML_PATH)
    cfg_paths = cfg['paths']
    
    # 경로 불러오기
    diffusion_results_path = cfg_paths['diffusion_results_path']
    duke_data_dir = cfg_paths['duke_data_dir']
    
    
    # 하이퍼파라미터 불러오기
    cfg_params = cfg['params'] 
    img_size = cfg_params['img_size']
    channels = cfg_params['channels']
    dataloader_batch_size = cfg_params['batch_size']
    learning_rate = cfg_params['learning_rate']
    timesteps = cfg_params['timesteps']

    # 결과 이미지 폴더 생성
    results_folder = Path(diffusion_results_path)
    results_folder.mkdir(exist_ok = True)

    # 데이터로더 생성
    dataloader = get_duke_dataloader(duke_data_dir, 
                                     dataloader_batch_size, 
                                     img_size, 
                                     cfg_params['gpu_num'] * cfg_params['base_num_workers']
                                     )                         

    # 모델 객체 생성
    diffusion_model = DiffusionUtils(total_timesteps=timesteps)
    model = Unet(
        dim=img_size,
        channels=channels,
        dim_mults=(1, 2, 4)
    )
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model, device_ids=GlobalVar.device_ids) 
    model.to(device=GlobalVar.device)
    
    # 옵티마이저 생성
    optimizer = Adam(model.parameters(), lr=learning_rate)
    # optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=0.9)
    
    # LR Scheduler 생성
    # NOTE: LR Scheduler는 다른거 테스트 후에 적용(이미 적용해보았는데, 큰 변화 없음)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer,
                                            lr_lambda=lambda epoch: 0.95 ** epoch,
                                            last_epoch=-1,
                                            verbose=False)
    
    # 학습 상태 디버그 용도 코드
    loss_list = list()
    
    train(
        cfg, 
        optimizer, 
        model, 
        diffusion_model, 
        dataloader,
        loss_list
        )
    


if __name__ == '__main__':    
    main()
