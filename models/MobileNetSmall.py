import torch.nn as nn
import torch 
import torchvision.models as models


class MobileNetSmall(nn.Module):
    def __init__(self):
        super(MobileNetSmall, self).__init__()
        self.model = models.mobilenet_v3_small(weights=None)
        self.model.classifier[3] = nn.Linear(self.model.classifier[3].in_features, 2)

        

    def forward(self, x):
        x = self.model(x)
        return x
    
if __name__ == '__main__':
    model = MobileNetSmall()
    print(model)