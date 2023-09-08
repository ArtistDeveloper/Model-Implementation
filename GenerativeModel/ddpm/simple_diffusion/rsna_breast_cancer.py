import os
import torch
import numpy as np

from torch.utils.data import Dataset
from torchvision import transforms
from skimage.io import imread

class RSNADataset(Dataset):
    def __init__(self, root_dir, img_size, is_resize = False):
        self.is_resize = is_resize
        self.data_dir = root_dir
        self.img_size = img_size
        self.img_name = [f for f in os.listdir(self.data_dir) if f.endswith('.png')]
        

    def normalize(self, img: np.ndarray):
        # 이미지의 각 픽셀 값을 범위[0, 255]로 normalize
        img = img.astype(float) * 255. / img.max() # uint16 -> float 범위 0~255로 정규화
        img = img.astype(np.uint8)

        return img


    def __getitem__(self, index):
        img_path = os.path.join(self.data_dir, self.img_name[index])

        img_arr = imread(img_path, as_gray=True)
        img_arr = self.normalize(img_arr)

        # ndarray를 텐서로 변환 및 float로 타입캐스팅
        data = torch.from_numpy(img_arr)
        data = data.type(torch.FloatTensor) 

        data = torch.unsqueeze(data, 0) # data 텐서의 0번째 차원에 크기 1의 새로운 차원 추가

        if self.is_resize:
            data = transforms.Resize((self.img_size, self.img_size))(data) # Resize 객체를 만든 뒤 __call__함수 호출
        
        return data
    

    def __len__(self):
        return len(self.img_name)