import torch

from Model_Implementation.GenerativeModel.dataset_class.rsna_breast_cancer import RSNADataset
from torchvision import transforms
 
IMG_SIZE = 64

img_transform = transforms.Compose(
    [
        transforms.Resize(size=(IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t * 2) - 1),  # NOTE: Scale between [-1, 1]
    ]
)

data_dir = r"/workspace/rsna_data"

dataset = RSNADataset(data_dir, img_transform)
data = next(iter(dataset))