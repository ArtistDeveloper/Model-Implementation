import os
import torch
import numpy as np

from torch.utils.data import Dataset
from torchvision import transforms
from skimage.io import imread


class RSNADataset(Dataset):
    def __init__(self, root_dir, transform=None):
        # self.is_resize = is_resize
        self.data_dir = root_dir
        self.img_names = [f for f in os.listdir(self.data_dir) if f.endswith(".png")]
        self.transform = transform if transform is not None else None

    def normalize(self, img: np.ndarray):
        # 이미지의 각 픽셀 값을 범위[0, 255]로 normalize
        img = img.astype(float) * 255.0 / img.max()  # uint16 -> float 범위 0~255로 정규화
        img = img.astype(np.uint8)

        return img

    def __getitem__(self, index):
        img_path = os.path.join(self.data_dir, self.img_names[index])

        img_arr = imread(img_path, as_gray=True)
        img_arr = self.normalize(img_arr)

        if self.transform is not None:
            data = self.transform(img_arr)
        else:
            # ndarray를 텐서로 변환 및 float로 타입캐스팅
            data = torch.from_numpy(img_arr)
            data = torch.unsqueeze(data, 0)  # data 텐서의 0번째 차원에 크기 1의 새로운 차원 추가
            data = data.type(torch.FloatTensor)

        return data

    def __len__(self):
        return len(self.img_names)
