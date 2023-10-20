import torch
from torchvision.utils import save_image
from tqdm.auto import tqdm

import ml_util
from Model_Implementation.GenerativeModel.ddpm.origin_ddpm.ddpm_for_breast import (
    Unet, Diffusion, ForwardBetaSchedule, SinusoidalPositionEmbeddings, ResnetBlock, Block,
    WeightStandardizedConv2d, Residual, PreNorm, LinearAttention, Attention
    )

def sample_image(model, difusion_frame, timestep, device, img_size, img_save_path):
    """
    마지막 timestep과 노이즈가 모두 디노이징 된 이미지를 저장합니다.
    """
    
    total_imgs = list()
    extracted_img = list()
    x_img = torch.randn((1, 1, img_size, img_size), device=device) # 정규분포의 형태의 랜덤 노이즈 생성
    
    for i in tqdm(reversed(range(0, timestep)), desc='sampling loop time step', total=timestep):
        x_img = difusion_frame.p_sample(model, x_img, torch.full((1,), i, device=device, dtype=torch.long), i)
        total_imgs.append(x_img)
        
    extracted_img = torch.cat([total_imgs[0], total_imgs[timestep-1]], dim=0)
    extracted_img = (extracted_img + 1) * 0.5 # norimalize -1~1 to 0~1
    save_image(extracted_img, str(img_save_path), nrow=1)


if __name__ == '__main__':
    TIMESTEP = 700
    model_path = r"/workspace/Model_Implementation/GenerativeModel/ddpm/origin_ddpm/saved_model/80_full_model.pth"
    img_save_path = r"./result.png"
    
    device = 'cuda:4'
    model = torch.load(model_path, map_location=device)
    diffusion_frame = Diffusion(total_timesteps=TIMESTEP)
    
    sample_image(model, diffusion_frame, TIMESTEP, device, 64, img_save_path)
    