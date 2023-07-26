from CNN.Resnet18.DukeClassification.dataset.DukeBreastDataset import DukeDataset

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def get_duke_dataloader(png_dir, train_batchsize=200, eval_batchsize = 10):
    dataset = DukeDataset(png_data_dir=png_dir, img_size=128)

    train_fraction = 0.8
    validation_fraction = 0.1
    test_fraction = 0.1
    dataset_size = len(dataset)

    num_train = int(train_fraction * dataset_size)
    num_validation = int(validation_fraction * dataset_size)
    num_test = int(test_fraction * dataset_size)

    train_dataset, validation_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [num_train, num_validation, num_test]
    )    

    train_loader = DataLoader(train_dataset, batch_size=train_batchsize, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=eval_batchsize)
    test_loader = DataLoader(test_dataset, batch_size=eval_batchsize)

    return train_loader, validation_loader, test_loader


def main():
    # GPUs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('running on {}'.format(device))

    PNG_DIR = r"C:\Users\JunsuPark\Desktop\Study\MachineLearning\ImplementingModel\CNN\Resnet18\DukeClassification\dataset\png_out"
    train_loader, validation_loader, test_loader = get_duke_dataloader(PNG_DIR, train_batchsize=200, eval_batchsize = 10)
    # print(len(train_loader.dataset), len(validation_loader.dataset), len(test_loader.dataset))

    



if __name__ == '__main__':
    main()