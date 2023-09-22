import os
import torch
import numpy as np

from torch.utils.data import Dataset
from torchvision import transforms
from skimage.io import imread
from PIL import Image


class RSNADataset(Dataset):
    def __init__(self, root_dir, transform=None):
        # self.is_resize = is_resize
        self.data_dir = root_dir
        self.img_names = [f for f in os.listdir(self.data_dir) if f.endswith(".png")]
        self.transform = transform if transform is not None else None


    def __getitem__(self, index):
        img_path = os.path.join(self.data_dir, self.img_names[index])
        img_arr = Image.open(img_path)
        
        if self.transform is not None:
            data = self.transform(img_arr)
        else:
            raise Exception("Transform does not exist.")

        return data

    def __len__(self):
        return len(self.img_names)
