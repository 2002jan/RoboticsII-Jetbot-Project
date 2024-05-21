import torch
import torch.nn as nn


class SteeringCNN(nn.Module):
    def __init__(self):
        super(SteeringCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.forward_layers = nn.Sequential(
            nn.Linear(25088, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        self.left_layers = nn.Sequential(
            nn.Linear(25088, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = torch.flatten(x, 1)
        forward = self.forward_layers(x)
        left = self.left_layers(x)
        return torch.cat((forward, left), dim=1)
