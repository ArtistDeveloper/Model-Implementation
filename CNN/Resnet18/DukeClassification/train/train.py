import os

from CNN.Resnet18.DukeClassification.dataset.duke_breast_ataset import DukeDataset
from CNN.Resnet18.DukeClassification.model.resnet18 import ResNet, ResNet18

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torchsummary import summary

from tqdm import tqdm

import matplotlib.pyplot as plt


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

    net = ResNet18()
    net = net.to(device)
    # print(summary(net, input_size=(1,128,128)))
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

    learning_rate = 0.1
    file_name = 'resnet18_cifar10.pt'

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0002)

    def train(epoch):
        print('\n[ Train epoch: %d]' % epoch)
        net.train()
        train_loss = 0
        correct = 0
        total_data_num = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            # optimizer.zero_grad()
            for param in net.parameters():
                param.grad = None

            benign_outputs = net(inputs)
            loss = criterion(benign_outputs, targets)
            loss.backward()
            
            optimizer.step()
            train_loss += loss.item()
            _, predicted_idx = benign_outputs.max(1)

            total_data_num += targets.size(0) # 128
            correct += predicted_idx.eq(targets).sum().item() # 예측이 맞은 데이터의 개수를 correct에 더함

            if batch_idx % 100 == 0:
                print('\nCurrent batch:', str(batch_idx))
                print('Current benign train accuracy:', str(predicted_idx.eq(targets).sum().item() / targets.size(0)))
                print('Current benign train loss:', loss.item())
        
        total_train_accuracy = correct / total_data_num * 100.
        print('\nTotal benign train accuracy:', total_train_accuracy)
        print('Total benign train loss:', train_loss)

        return total_train_accuracy


    def test(epoch):
        print('\n[ Test epoch: %d]' % epoch)
        net.eval()
        loss = 0
        correct = 0
        total_data_num = 0

        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            total_data_num += targets.size(0)

            outputs = net(inputs)
            loss += criterion(outputs, targets).item()

            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()


        total_test_accuracy = correct / total_data_num * 100.
        print('\nTest acuracy:', total_test_accuracy)
        print('Test average loss:', loss / total_data_num)

        state = {
            'net': net.state_dict()
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/' + file_name)
        print('Model Saved!')

        return total_test_accuracy


    def adjust_learning_rate(optimizer, epoch): # Learning rate scheduler
        lr = learning_rate
        if epoch >= 100:
            lr /= 10
        if epoch >= 150:
            lr /= 10
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    train_accuracies = []
    test_accuracies = []

    for epoch in tqdm(range(0, 200)):
        adjust_learning_rate(optimizer, epoch)
        train_accuracies.append(train(epoch))
        test_accuracies.append(test(epoch))

    plt.xlabel('epoch')
    plt.ylabel('acc')
    plt.plot(range(1, 201), train_accuracies, label='Train Accuracy')
    plt.plot(range(1, 201), test_accuracies, label='Test Accuracy')
    plt.show()
    

if __name__ == '__main__':
    main()