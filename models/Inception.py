import torch

import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=False):
        super(ConvBlock, self).__init__()

        self.conv2d = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                padding=padding, stride=stride, bias=False)

        self.batchnorm2d = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.batchnorm2d(self.conv2d(x)))


class InceptionBlock(nn.Module):
    def __init__(self, in_channels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_1x1_pooling):
        super(InceptionBlock, self).__init__()

        self.branch1 = ConvBlock(in_channels, out_1x1, 1, 1, 0)

        self.branch2 = nn.Sequential(
            ConvBlock(in_channels, red_3x3, 1, 1, 0),
            ConvBlock(red_3x3, out_3x3, 3, 1, 1)
        )

        self.branch3 = nn.Sequential(
            ConvBlock(in_channels, red_5x5, 1, 1, 0),
            ConvBlock(red_5x5, out_5x5, 5, 1, 2)
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            ConvBlock(in_channels, out_1x1_pooling, 1, 1, 0)
        )

    def forward(self, x):
        return torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], dim=1)


class Inception(nn.Module):
    def __init__(self, in_channels):
        super(Inception, self).__init__()

        self.conv1 = ConvBlock(in_channels, 64, 7, 2, 3)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2 = nn.Sequential(
            ConvBlock(64, 64, 1, 1, 0),
            ConvBlock(64, 192, 3, 1, 1)
        )
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception3a = InceptionBlock(192, 64, 96, 128, 16, 32, 32)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception4a = InceptionBlock(256, 128, 96, 196, 16, 24, 32)

        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)

        self.fc1 = nn.Sequential(
            nn.Linear(24320, 64),
            nn.ReLU()
        )

        self.fc_forward = nn.Linear(64, 1)
        self.fc_left = nn.Linear(64, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.maxpool2(x)

        x = self.inception3a(x)
        x = self.maxpool3(x)

        x = self.inception4a(x)
        x = self.avgpool(x)

        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        forward = F.sigmoid(self.fc_forward(x))
        left = F.tanh(self.fc_left(x))

        return torch.cat((forward, left), dim=1)
