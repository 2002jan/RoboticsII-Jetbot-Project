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
    def __init__(self, in_channels, out_1x1, out_3x3):
        super(InceptionBlock, self).__init__()

        self.branch1 = ConvBlock(in_channels, out_1x1, 1, 1, 0)

        self.branch2 = ConvBlock(in_channels, out_3x3, 3, 1, 1)

    def forward(self, x):
        return torch.cat([self.branch1(x), self.branch2(x)], dim=1)


class DownsampleBlock(nn.Module):
    def __init__(self, in_channels, out_3x3):
        super(DownsampleBlock, self).__init__()

        self.branch1 = ConvBlock(in_channels, out_3x3, 7, 5, 0)

        self.branch2 = nn.MaxPool2d(kernel_size=7, stride=5, padding=0)

    def forward(self, x):
        return torch.cat([self.branch1(x), self.branch2(x)], dim=1)


class SmallInception(nn.Module):
    def __init__(self, in_channels):
        super(SmallInception, self).__init__()

        self.conv1 = ConvBlock(in_channels, 48, 3, 1, 1)

        self.inception1 = nn.Sequential(
            InceptionBlock(48, 8, 24),
            InceptionBlock(32, 8, 32),
            DownsampleBlock(40, 40)
        )

        self.inception2 = nn.Sequential(
            InceptionBlock(80, 32, 48),
            InceptionBlock(80, 32, 48),
            InceptionBlock(80, 32, 48),
            InceptionBlock(80, 16, 48),
            DownsampleBlock(64, 48)
        )

        self.inception3 = nn.Sequential(
            InceptionBlock(112, 72, 96),
            InceptionBlock(168, 72, 96),
            nn.AvgPool2d(7, 7, 0)
        )

        self.fc1 = nn.Sequential(
            nn.Linear(168, 64),
            nn.ReLU()
        )

        self.fc_forward = nn.Linear(64, 1)
        self.fc_left = nn.Linear(64, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.inception1(x)
        x = self.inception2(x)
        x = self.inception3(x)

        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        forward = F.sigmoid(self.fc_forward(x))
        left = F.tanh(self.fc_left(x))

        return torch.cat((forward, left), dim=1)
