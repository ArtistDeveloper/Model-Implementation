import os
import torch

from ddpm import Unet
from ddpm import SinusoidalPositionEmbeddings
from ddpm import ResnetBlock
from ddpm import Block
from ddpm import WeightStandardizedConv2d
from ddpm import Residual
from ddpm import PreNorm
from ddpm import LinearAttention
from ddpm import Attention

from ddpm import Diffusion

import matplotlib.pyplot as plt
import matplotlib.animation as animation

MODEL_PATH = "/workspace/Model-Implementation/GenerativeModel/ddpm/saved_model/ddpm.pt"
TIMESTEPS=100

# load model
model = torch.load(MODEL_PATH)
model.eval()

diffusion_model = Diffusion(total_timesteps=TIMESTEPS)
samples = diffusion_model.sample(model, image_size=28, batch_size=64, channels=1)

# show a random one
random_index = 5
plt.imshow(samples[-1][random_index].reshape(28, 28, 1), cmap="gray")

random_index = 53

fig = plt.figure()
ims = []
for i in range(TIMESTEPS):
    im = plt.imshow(samples[i][random_index].reshape(28, 28, 1), cmap="gray", animated=True)
    ims.append([im])

animate = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)
animate.save('diffusion.gif')
# plt.show()
plt.savefig("./result.png")