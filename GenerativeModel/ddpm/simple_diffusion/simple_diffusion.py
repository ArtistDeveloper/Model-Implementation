import torch
import torchvision
import torch.nn.functional as F
from torchvision import transforms

import matplotlib.pyplot as plt


from Model_Implementation.GenerativeModel.ddpm.simple_diffusion.rsna_breast_cancer import (
    RSNADataset,
)


def show_images(dataset, num_samples=20, cols=4):
    """Plots some samples from the dataset"""
    plt.figure(figsize=(15, 15))

    for i, img in enumerate(dataset):
        if i == num_samples:
            break
        plt.subplot(int(num_samples / cols) + 1, cols, i + 1)
        plt.imshow(img[0])

    plt.savefig("./test.png")


# Forward process
def linear_beta_schedule(timesteps, start=0.0001, end=0.02):
    return torch.linspace(start, end, timesteps)


def get_index_from_list(vals, t, x_shape):
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


def forward_diffusion_sample(x_0, t, device="cpu"):
    noise = torch.randn_like(x_0)
    sqrt_alphas_cumprod_t = get_index_from_list(sqrt_alphas_cumprod, t, x_0.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x_0.shape
    )

    # mean + variance
    return sqrt_alphas_cumprod_t.to(device) * x_0.to(device) \
    + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device), noise.to(device)


T = 300
betas = linear_beta_schedule(timesteps=T)

# Pre-calculate different terms for closed form
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)



def main():
    img_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(size=(64, 64)),
        ]
    )

    dataset = RSNADataset("/workspace/rsna_data/", img_transform)
    show_images(dataset)


if __name__ == "__main__":
    main()





# # In[3]:


# from torchvision import transforms
# from torch.utils.data import DataLoader
# import numpy as np

# IMG_SIZE = 64
# BATCH_SIZE = 24


# def show_tensor_image(image):
#     reverse_transforms = transforms.Compose(
#         [
#             transforms.Lambda(lambda t: (t + 1) / 2),
#             transforms.Lambda(lambda t: t.permute(1, 2, 0)),  # CHW to HWC
#             transforms.Lambda(lambda t: t * 255.0),
#             transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
#             transforms.ToPILImage(),
#         ]
#     )

#     # Take first image of batch
#     if len(image.shape) == 4:
#         image = image[0, :, :, :]
#     plt.imshow(reverse_transforms(image), cmap="gray")  # NOTE: origin: cmap=gray


# # data = RSNADataset('/workspace/rsna_data/', IMG_SIZE, is_resize=True)
# # dataloader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

# data = torchvision.datasets.Flowers102(
#     root="./",
#     download=True,
#     transform=transforms.Compose(
#         [
#             transforms.Resize((IMG_SIZE, IMG_SIZE)),
#             transforms.ToTensor(),
#             transforms.Lambda(lambda t: (t * 2) - 1),  # NOTE: Scale between [-1, 1]
#         ]
#     ),
# )
# dataloader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)


# # In[4]:


# # Simulate forward diffusion
# image = next(iter(dataloader))[0]

# plt.figure(figsize=(20, 20))
# plt.axis("off")
# num_images = 10
# stepsize = int(T / num_images)

# for idx in range(0, T, stepsize):
#     t = torch.Tensor([idx]).type(torch.int64)
#     plt.subplot(1, num_images + 1, int(idx / stepsize) + 1)
#     img, noise = forward_diffusion_sample(image, t)
#     show_tensor_image(img)


# # # ### The backward process = U-Net

# # # In[5]:


# # """
# # 이 컨볼루션 레이어에 학습 가능한 필터를 적용하여 채널을 확장합니다.

# # 그래서 점점 더 많은 채널을 갖게 되고 텐서의 깊이가 증가합니다.

# # 1024개 채널이 될 때까지 이런 과정을 거친 다음 위쪽 채널에서 다시 크기를 줄입니다.

# # 그 외에도 이미지의 크기를 늘리고 줄이면 이 모든 것이 이 블록에서 일어나는 것을 잠시 후에 볼 수 있습니다.

# # 앞서 언급했듯이 위치 벡터의 형태로 위치 임베딩을 사용하며 여기서는 사인 및 코사인 함수를 사용하여 계산하는 방법이 있습니다.
# # 그리고 이것은 목록에서 인덱스의 위치를 설명하는 벡터를 반환합니다.


# # 또한 이미지를 첫 번째 차원으로 변환하여 64개의 채널로 변환하는 초기 투영 레이어와 마지막 차원을 다시 64개로 사용하여 이미지의 채널 수인 3개로 다시 변환하는 판독 레이어가 있습니다.


# # block에 대한 설명: 영상 참조하여 추가 작성


# # DDPM의 appendix B의 Experimental details를 확인하면 block관련된 내용을 확인할 수 있다.

# # """

# # from torch import nn
# # import math


# # class Block(nn.Module):
# #     def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
# #         super().__init__()
# #         self.time_mlp = nn.Linear(time_emb_dim, out_ch)
# #         if up:
# #             self.conv1 = nn.Conv2d(2 * in_ch, out_ch, 3, padding=1)
# #             self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
# #         else:
# #             self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
# #             self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)
# #         self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
# #         self.bnorm1 = nn.BatchNorm2d(out_ch)
# #         self.bnorm2 = nn.BatchNorm2d(out_ch)
# #         self.relu = nn.ReLU()

# #     def forward(
# #         self,
# #         x,
# #         t,
# #     ):
# #         # First Conv
# #         h = self.bnorm1(self.relu(self.conv1(x)))
# #         # Time embedding
# #         time_emb = self.relu(self.time_mlp(t))
# #         # Extend last 2 dimensions
# #         time_emb = time_emb[(...,) + (None,) * 2]
# #         # Add time channel
# #         h = h + time_emb
# #         # Second Conv
# #         h = self.bnorm2(self.relu(self.conv2(h)))
# #         # Down or Upsample
# #         return self.transform(h)


# # class SinusoidalPositionEmbeddings(nn.Module):
# #     def __init__(self, dim):
# #         super().__init__()
# #         self.dim = dim

# #     def forward(self, time):
# #         device = time.device
# #         half_dim = self.dim // 2
# #         embeddings = math.log(10000) / (half_dim - 1)
# #         embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
# #         embeddings = time[:, None] * embeddings[None, :]
# #         embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
# #         # TODO: Double check the ordering here
# #         return embeddings


# # class SimpleUnet(nn.Module):
# #     """
# #     모든 모델에서는 해상도의 레벨 당 두 개의 컨볼루셔널 잔차 블록과 16x16의 셀프 어텐션 블록을 가진다.
# #     여기서는 simple하게 해상도의 레벨 당 하나의 convolutional 블록을 가지는 듯 하다.
# #     """

# #     def __init__(self):
# #         super().__init__()
# #         image_channels = 3  # NOTE: origin: 1
# #         down_channels = (64, 128, 256, 512, 1024)
# #         up_channels = (1024, 512, 256, 128, 64)
# #         out_dim = 3  # NOTE: origin: 1
# #         time_emb_dim = 32

# #         # Time embedding
# #         self.time_mlp = nn.Sequential(
# #             SinusoidalPositionEmbeddings(time_emb_dim),
# #             nn.Linear(time_emb_dim, time_emb_dim),
# #             nn.ReLU(),
# #         )

# #         # Initial projection
# #         self.conv0 = nn.Conv2d(image_channels, down_channels[0], 3, padding=1)

# #         # Downsample
# #         self.downs = nn.ModuleList(
# #             [
# #                 Block(down_channels[i], down_channels[i + 1], time_emb_dim)
# #                 for i in range(len(down_channels) - 1)
# #             ]
# #         )
# #         # Upsample
# #         self.ups = nn.ModuleList(
# #             [
# #                 Block(up_channels[i], up_channels[i + 1], time_emb_dim, up=True)
# #                 for i in range(len(up_channels) - 1)
# #             ]
# #         )

# #         # Edit: Corrected a bug found by Jakub C (see YouTube comment)
# #         self.output = nn.Conv2d(up_channels[-1], out_dim, 1)

# #     def forward(self, x, timestep):
# #         # Embedd time
# #         t = self.time_mlp(timestep)
# #         # Initial conv
# #         x = self.conv0(x)
# #         # Unet
# #         residual_inputs = []
# #         for down in self.downs:
# #             x = down(x, t)
# #             residual_inputs.append(x)
# #         for up in self.ups:
# #             residual_x = residual_inputs.pop()
# #             # Add residual x as additional channels
# #             x = torch.cat((x, residual_x), dim=1)
# #             x = up(x, t)
# #         return self.output(x)


# # model = SimpleUnet()
# # print("Num params: ", sum(p.numel() for p in model.parameters()))
# # model


# # # ### Loss

# # # In[6]:


# # def get_loss(model, x_0, t):
# #     x_noisy, noise = forward_diffusion_sample(x_0, t, device)
# #     noise_pred = model(x_noisy, t)
# #     return F.l1_loss(noise, noise_pred)


# # # ### Sampling

# # # In[7]:


# # @torch.no_grad()
# # def sample_timestep(x, t):
# #     """
# #     Calls the model to predict the noise in the image and returns
# #     the denoised image.
# #     Applies noise to this image, if we are not in the last step yet.
# #     """
# #     betas_t = get_index_from_list(betas, t, x.shape)
# #     sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
# #         sqrt_one_minus_alphas_cumprod, t, x.shape
# #     )
# #     sqrt_recip_alphas_t = get_index_from_list(sqrt_recip_alphas, t, x.shape)

# #     # Call model (current image - noise prediction)
# #     model_mean = sqrt_recip_alphas_t * (
# #         x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
# #     )
# #     posterior_variance_t = get_index_from_list(posterior_variance, t, x.shape)

# #     if t == 0:
# #         # As pointed out by Luis Pereira (see YouTube comment)
# #         # The t's are offset from the t's in the paper
# #         return model_mean
# #     else:
# #         noise = torch.randn_like(x)
# #         return model_mean + torch.sqrt(posterior_variance_t) * noise


# # @torch.no_grad()
# # def sample_plot_image():
# #     # Sample noise
# #     img_size = IMG_SIZE
# #     channel_size = 3

# #     img = torch.randn((1, channel_size, img_size, img_size), device=device)
# #     plt.figure(figsize=(15, 15))
# #     plt.axis("off")
# #     num_images = 10
# #     stepsize = int(T / num_images)

# #     for i in range(0, T)[::-1]:
# #         t = torch.full((1,), i, device=device, dtype=torch.long)
# #         img = sample_timestep(img, t)
# #         # Edit: This is to maintain the natural range of the distribution
# #         img = torch.clamp(img, -1.0, 1.0)
# #         if i % stepsize == 0:
# #             plt.subplot(1, num_images, int(i / stepsize) + 1)
# #             show_tensor_image(img.detach().cpu())
# #     plt.show()


# # # ### Training

# # # In[8]:


# # from torch.optim import Adam

# # MODEL_PATH = "/workspace/Model_Implementation/GenerativeModel/ddpm/simple_diffusion/simple_diff.pt"
# # device = "cuda:3" if torch.cuda.is_available() else "cpu"
# # model.to(device)
# # optimizer = Adam(model.parameters(), lr=0.001)
# # epochs = 500  # Try more!

# # for epoch in range(epochs):
# #     for step, batch in enumerate(dataloader):
# #         optimizer.zero_grad()

# #         t = torch.randint(0, T, (BATCH_SIZE,), device=device).long()
# #         # loss = get_loss(model, batch, t)
# #         loss = get_loss(model, batch[0], t)

# #         loss.backward()
# #         optimizer.step()

# #         if epoch % 5 == 0 and step == 0:
# #             print(f"Epoch {epoch} | step {step:03d} Loss: {loss.item()} ")
# #             sample_plot_image()


# # torch.save(model, MODEL_PATH)


# # # In[ ]:


# # # load model
# # model = torch.load(MODEL_PATH)
# # model.eval()

# # sample_plot_image()


# # # In[ ]:


# # # In[ ]:

# # %%