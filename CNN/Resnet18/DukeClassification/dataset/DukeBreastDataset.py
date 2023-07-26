import os
import torch
import numpy as np

from torch.utils.data import Dataset
from torchvision import transforms
from skimage.io import imread

class DukeDataset(Dataset):
    def __init__(self, png_data_dir, img_size):
        self.data_dir = png_data_dir
        self.img_size = img_size

        self.labels = None
        self.create_labels()


    def create_labels(self):
        # 레이블 생성 및 저장 (각 이미지에서 neg를 0으로, pos를 1로 target_label할당)
        # 각 레이블은 튜플 (이미지 파일 이름, 레이블 번호)

        path_with_labels = []
        print('building Duke dataset labels.')

        for target_index, target_label in enumerate(['neg', 'pos']):
            case_dir = os.path.join(self.data_dir, target_label)

            for fname in os.listdir(case_dir):
                if '.png' in fname:
                    file_path = os.path.join(case_dir, fname)
                    path_with_labels.append((file_path, target_index))

        self.labels = path_with_labels


    def normalize(self, img: np.ndarray):
        # 이미지의 각 픽셀 값을 범위[0, 255]로 normalize

        img = img.astype(float) * 255. / img.max() # uint16 -> float 범위 0~255로 정규화
        img = img.astype(np.uint8)

        return img
    
    def __getitem__(self, index):
        file_path, target = self.labels[index]

        img_arr = imread(file_path, as_gray=True)
        img_arr = self.normalize(img_arr)

        # ndarray를 텐서로 변환 및 float로 타입캐스팅
        data = torch.from_numpy(img_arr)
        data = data.type(torch.FloatTensor) 

        data = torch.unsqueeze(data, 0) # data 텐서의 0번째 차원에 크기 1의 새로운 차원 추가
        data = transforms.Resize((self.img_size, self.img_size))(data) # Resize 객체를 만든 뒤 __call__함수 호출
        
        return data, target
    

    def __len__(self):
        return len(self.labels)