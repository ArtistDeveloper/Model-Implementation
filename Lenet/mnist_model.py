import torch.nn as nn

class Lenet5(nn.Module):
    def __init__(self):
        super().__init__()

        self.Conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5),
            nn.Tanh()
        )

        self.Subsampling1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

        self.Conv2 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.Tanh()
        )

        self.Subsampling2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

        # 논문을 읽어보면 이미지에서는 C5는 fully connected layer같아 보이지만,
        # 논문 8페이지 C5 Layer에 대해 읽어보면 convolutional layer라는 것을 알 수 있다.
        # 5x5 kernel size를 통해 feature size를 1x1로 만든다.
        self.Conv3 = nn.Sequential( 
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1),
            nn.Tanh()
        )

        self.fc1 = nn.Sequential(
            nn.Linear(in_features=120, out_features=84),
            nn.Tanh()
        )

        self.fc2 = nn.Sequential(
            nn.Linear(in_features=84, out_features=10),
            nn.Tanh()
        )

        
    def forward(self, x):
        x = self.Conv1(x)
        x = self.Subsampling1(x)
        x = self.Conv2(x)
        x = self.Subsampling2(x)
        x = self.Conv3(x)
        x = x.view(-1, 120)
        x = self.fc1(x)
        x = self.fc2(x)

        return x

