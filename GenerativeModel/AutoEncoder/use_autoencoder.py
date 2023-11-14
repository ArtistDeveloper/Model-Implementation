import torch
import matplotlib.pyplot as plt
import numpy as np

from autoencoder import AutoEncoder
from autoencoder import draw_decoder_output


if __name__ == '__main__':
    MODEL_SAVE_PATH = \
        r"/workspace/Model_Implementation/GenerativeModel/AutoEncoder/autoencoder_models/autoencoder.pt"
    RESULT_SAVE_PATH = \
        r"/workspace/Model_Implementation/GenerativeModel/AutoEncoder/autoencoder_results/test_img.png"    
    
    
    device = torch.device('cuda')
    autoencoder = AutoEncoder()
    autoencoder.load_state_dict(torch.load(MODEL_SAVE_PATH))
    autoencoder.to(device)
    
    # rand_noise = torch.randn((28, 28))
    rand_noise = torch.rand((28, 28))
    rand_noise = rand_noise.view(-1, 28*28).to(device)
    
    encoded_data, decoded_data = autoencoder(rand_noise)
    
    fig, ax = plt.subplots(3, 1, figsize=(5, 5), squeeze=False)

    # 입력 이미지
    img = np.reshape(rand_noise.cpu().numpy(), (28, 28))
    ax[0][0].imshow(img, cmap='gray')
    
    # 인코딩 이미지
    img = np.reshape(encoded_data.detach().cpu().numpy(), (1, 3))
    print("IMG: ", img)
    ax[1][0].imshow(img, cmap='gray')
    
    # 생성된 이미지
    img = np.reshape(decoded_data.data.cpu().numpy(), (28, 28))
    ax[2][0].imshow(img, cmap='gray')
    
    
    plt.savefig(RESULT_SAVE_PATH)

    
    