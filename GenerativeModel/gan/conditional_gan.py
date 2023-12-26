import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision import utils

from torch.utils.data import DataLoader
from torch import optim

import numpy as np
import matplotlib.pyplot as plt



class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_classes = 10
        self.nz = 100
        self.input_size = (1, 28, 28)
        
        # Noise와 label을 결합하는 용도인 label embedding matrix를 생성
        # 해당 embedding 값 또한 학습 가능한 파라미터임에 유의.
        self.label_emb = nn.Embedding(self.num_classes, self.num_classes) # num embedding, embedding_dim
        
        # Generator
        self.gen = nn.Sequential(
            nn.Linear(self.nz + self.num_classes, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256,512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512,1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, int(np.prod(self.input_size))), # 1024, 784
            nn.Tanh()
        )
    
    def forward(self, noise, labels):
        # noise와 label의 결합     
        gen_input = torch.cat((self.label_emb(labels), noise), -1)
        x = self.gen(gen_input)
        x = x.view(x.size(0), *self.input_size)
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_size = (1, 28, 28)
        self.num_classes = 10
        self.label_emb = nn.Embedding(self.num_classes, self.num_classes)
        self.dis = nn.Sequential(
            nn.Linear(self.num_classes + int(np.prod(self.input_size)), 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512,512),
            nn.Dropout(0.4),
            
            nn.LeakyReLU(0.2),
            nn.Linear(512,512),
            nn.Dropout(0.4),
            
            nn.LeakyReLU(0.2),
            nn.Linear(512,1),
            nn.Sigmoid()
        )
        
    def forward(self, img, labels):
        dis_input = torch.cat((img.view(img.size(0), -1), self.label_emb(labels)), -1)
        x = self.dis(dis_input)
        return x
    

def initialize_weights(model):
    classname = model.__class__.__name__
    # fc layer
    if classname.find('Linear') != -1:
        nn.init.normal_(model.weight.data, 0.0, 0.02)
        nn.init.constant_(model.bias.data, 0)
    # batchnorm
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(model.weight.data, 1.0, 0.02)
        nn.init.constant_(model.bias.data, 0)


def check_data(train_dataset, img_save_path):
    img, label = train_dataset.data, train_dataset.targets
    
    # Make it to 4D Tensor1
    # 기존 : (#Batch) x (height) x (width) -> (#Batch) x (#channel) x (height) x(width)
    if len(img.shape) == 3:
        img = img.unsqueeze(1)
    
    # Visualize
    img_grid = utils.make_grid(img[:40], ncol=8, padding=2)
    show(img_grid, img_save_path)


def show(img, path):
    img = img.numpy() # Tensor -> numpy array
    img = img.transpose([1,2,0]) # C x H x W -> H x W x C
    plt.imshow(img, interpolation='nearest')
    plt.savefig(path + "/test.png")


def show_generated_image(model_gen, path, device):
    model_gen.eval()
    
    # fake image 생성
    with torch.no_grad():
        fig = plt.figure(figsize=(8,8))
        cols, rows = 4, 4 # row와 col 갯수
        for i in range(rows * cols):
            fixed_noise = torch.randn(16, 100, device=device)
            label = torch.randint(0,10,(16,), device=device)
            img_fake = model_gen(fixed_noise, label).detach().cpu()
            fig.add_subplot(rows, cols, i+1)
            plt.title(label[i].item())
            plt.axis('off')
            plt.imshow(img_fake[i].squeeze(), cmap='gray')
    plt.savefig(path + "/generated_img.png")


def train(model_dis, model_gen, train_loader, device):
    loss_func = nn.BCELoss()
    
    lr = 2e-4
    beta1 = 0.5
    beta2 = 0.999
    
    opt_dis = optim.Adam(model_dis.parameters(), lr=lr, betas=(beta1,beta2))
    opt_gen = optim.Adam(model_gen.parameters(), lr=lr, betas=(beta1,beta2))
    
    nz = 100
    num_epochs = 100
    
    loss_history = {'gen': [], 'dis': []}
    
    batch_count = 0
    start_time = time.time()
    model_dis.train()
    model_gen.train()
    
    for epoch in range(num_epochs):
        for x_batch, y_batch in train_loader:
            batch_size = x_batch.shape[0]
            
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            y_batch_real = torch.Tensor(batch_size, 1).fill_(1.0).to(device)
            y_batch_fake = torch.Tensor(batch_size, 1).fill_(0.0).to(device)
            
            # Generator 학습 시작
            model_gen.zero_grad()
            noise = torch.randn(batch_size, 100).to(device)
            gen_label = torch.randint(0, 10, (batch_size, )).to(device)
            
            # 가짜 이미지 생성
            generated_img = model_gen(noise, gen_label)
            
            # 가짜 이미지 판별
            dis_result = model_dis(generated_img, gen_label)
            
            # discriminator가 1에 가까운 출력을 낼 수 있도록 generator를 학습.
            loss_gen = loss_func(dis_result, y_batch_real)
            loss_gen.backward()
            opt_gen.step()
            
            # Discriminator 학습 시작
            model_dis.zero_grad()
            
            # 진짜 이미지 판별
            dis_result = model_dis(x_batch, y_batch)
            loss_real = loss_func(dis_result, y_batch_real)
            
            # 가짜 이미지 판별
            # Discriminator가 가짜이미지로 분류한 값과, y_batch_fake의 값의 차이를 줄임으로
            # 가짜이미지를 가짜이미지로 분류할 수 있는 성능을 올림
            out_dis = model_dis(generated_img.detach(), gen_label)
            loss_fake = loss_func(out_dis, y_batch_fake)
            
            # 진짜 이미지 판별 loss와 가짜 이미지 판별 loss를 더한 뒤 2를 나누어 loss값을 사용한다. (GAN loss를 구현할 떄는 이와 같은 방식을 따름)
            loss_dis = (loss_real + loss_fake) / 2
            loss_dis.backward()
            opt_dis.step()

            loss_history['gen'].append(loss_gen.item())
            loss_history['dis'].append(loss_dis.item())
            
            batch_count += 1
            if batch_count % 1000 == 0:
                print('Epoch: %.0f, G_Loss: %.6f, D_Loss: %.6f, time: %.2f min' %(epoch, loss_gen.item(), loss_dis.item(), (time.time()-start_time)/60))
                
    return model_dis, model_gen
            
    


def main():
    device = torch.device('cuda:5' if torch.cuda.is_available() else 'cpu')
    img_save_path = r"/workspace/Model_Implementation/GenerativeModel/gan/results"
    
    # Set Data path
    datapath = './data'
    os.makedirs(datapath, exist_ok=True)

    # Pre-process
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    # Laod MNIST
    train_dataset = datasets.MNIST(datapath, train=True, download=True, transform=trans)
    
    # Check data
    check_data(train_dataset, img_save_path)
    
    # DatLoader
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    print(len(train_loader))
    
    model_gen = Generator().to(device)
    model_dis = Discriminator().to(device)
    
    # Apply weight initialization
    model_gen.apply(initialize_weights);
    model_dis.apply(initialize_weights);
    
    model_dis, model_gen = train(model_dis, model_gen, train_loader, device)
    show_generated_image(model_gen, img_save_path, device)
    

if __name__=='__main__':
    main()