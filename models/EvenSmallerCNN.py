import torch
import torch.nn as nn


class EvenSmallerCNN(nn.Module):
    def __init__(self):
        super(EvenSmallerCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc_layers_forward = nn.Sequential(
            nn.Linear(128 * 7 * 7, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Tanh()
        )
        self.fc_layers_left = nn.Sequential(
            nn.Linear(128 * 7 * 7, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = torch.flatten(x, 1)
        forward = self.fc_layers_forward(x)
        left = self.fc_layers_left(x)
        return torch.cat((forward, left), dim=1)
