import os

from torch.utils.data import Dataset
from PIL import Image


class DukeDataset(Dataset):
    def __init__(self, data_dir, transform):
        self.data_dir = data_dir
        self.transform = transform if transform is not None else None
        self.imgs = list()
        self.combine_data()
    
    def combine_data(self):
        print("Combine Duke dataset labels")
        imgs = list()
        
        for target, target_label in enumerate(['neg', 'pos']):
            case_dir = os.path.join(self.data_dir, target_label)
            
            for fname in os.listdir(case_dir):
                if '.png' in fname:
                    file_path = os.path.join(case_dir, fname)
                    imgs.append((file_path, target))        
                    
        self.imgs = imgs
    
    def __getitem__(self, index):
        img_path = os.path.join(self.data_dir, self.imgs[index])
        img_arr = Image.open(img_path)
        
        if self.transform is not None:
            data = self.transform(img_arr)
        else:
            raise Exception("Transform does not exist.")

        return data

    def __len__(self):
        return len(self.imgs)
    
        
    