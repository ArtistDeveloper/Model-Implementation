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


# class AutoEncoderConv(nn.Module):
#     def __init__(self):
#         super(AutoEncoderConv, self).__init__()
        
#         self.encoder = nn.Sequential(
#             # 28x28 -> 14x14
#             nn.Conv2d(1, 32, 3, stride=1, padding=1),
#             nn.Conv2d(32, 64, 3, stride=2, padding=1),
#             nn.ReLU(),
            
#             # 14x14 -> 7x7
#             nn.Conv2d(64, 64, 3, stride=1, padding=1),
#             nn.Conv2d(64, 128, 3, stride=2, padding=1),
#             nn.ReLU(),
            
#             # 7x7 -> 3x3
#             nn.Conv2d(128, 128, 3, stride=1, padding=1),
#             nn.Conv2d(128, 256, 4, stride=2, padding=1),
#             nn.ReLU(),
#         )
        
#         self.decoder = nn.Sequential(
#             # 3x3 -> 7x7
#             nn.Conv2d(256, 256, 3, stride=1, padding=1),
#             nn.ConvTranspose2d(256, 128, 3, stride=2, padding=0, output_padding=0),
#             nn.ReLU(),
            
#             # 7x7 -> 14x14
#             nn.Conv2d(128, 64, 3, stride=1, padding=1),
#             nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
#             nn.ReLU(),
            
#             # 14x14 -> 28x28
#             nn.Conv2d(32, 32, 3, stride=1, padding=1),
#             nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
#             nn.ReLU(),
            
#             nn.Conv2d(16, 1, 3, stride=1, padding=1),
#             nn.Sigmoid()  # 변경된 부분
#         )
        
    
#     def forward(self, x):
#         encoded = self.encoder(x)
#         decoded = self.decoder(encoded)
#         return encoded, decoded


class AutoEncoderConv(nn.Module):
    # TODO: 28x28에서 돌아가도록 수정하기 -> 만약 뒤에도 안된다면 이건 데이터 전달이 잘못된지도 확인이 필요할지도..
    def __init__(self):
        super(AutoEncoderConv, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size = 2, stride = 2, padding=0),
            nn.ReLU(),
            
            nn.ConvTranspose2d(16, 3, kernel_size = 2, stride = 2, padding=0),
            nn.Sigmoid()
        )
        
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


def draw_decoder_output(origin_data, autoenocder, epoch, device): 
    test_x = origin_data.to(device)
    _, decoded_data = autoenocder(test_x)
    
    fig, ax = plt.subplots(2, 5, figsize=(5,2))
    print("[Epoch {}]".format(epoch))
    
    # 원본 데이터 출력
    for i in range(5):
        # img = np.reshape(origin_data.data.numpy()[i], (28, 28)) # 파이토치 텐서를 넘파이로 변환
        img = np.reshape(origin_data.data.numpy()[i], (28,28))
        ax[0][i].imshow(img, cmap='gray')
        ax[0][i].set_xticks(()); ax[0][i].set_yticks(())
    
    # 생성된 데이터 출력
    for i in range(5):
        img = np.reshape(decoded_data.data.cpu().numpy()[i], (28,28))
        ax[1][i].imshow(img, cmap='gray')
        ax[1][i].set_xticks(()); ax[0][i].set_yticks(())
    
    plt.savefig(f'/workspace/Model_Implementation/GenerativeModel/AutoEncoder/autoencoder_results/{epoch}_img.png')


def train(epoch, model, train_loader, device, optimizer, criterion, origin_data):
    model.train()
    
    for epoch in range(0, epoch):
        for step, (x, label) in enumerate(train_loader):
            x = x.to(device)
            y = x.clone().to(device) # x(입력)와 y(대상 레이블) 모두 원본이미지 x이다.
            label = label.to(device)
                        
            encoded, decoded = model(x)
            
            loss = criterion(decoded, y)
            if step % 100 == 0:
                print("loss: ", loss.item())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        draw_decoder_output(origin_data, model, epoch, device)    
            
    return model


def main():
    MODEL_SAVE_PATH = r"/workspace/Model_Implementation/GenerativeModel/AutoEncoder/autoencoder_models/autoencoder_conv.pt"
    epoch = 50
    batch_size = 64
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda:5' if use_cuda else 'cpu')
    
    trainset = datasets.FashionMNIST(
        root='./data/',
        train=True,
        download=True,
        transform=transforms.ToTensor()
    )
    
    train_loader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True)
    
    model = AutoEncoderConv().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.MSELoss()
    
    origin_data = trainset.data[:5].unsqueeze(dim=1)
    origin_data = origin_data.type(torch.FloatTensor) / 255.
    
    model = train(epoch, model, train_loader, device, optimizer, criterion, origin_data)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    


if __name__ == '__main__':
    main()