import torch
import torch.nn as nn
from torchvision import models

class ResNetClassifier(nn.Module):
    def __init__(self, num_classes):
        super(ResNetClassifier, self).__init__()
        self.base_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, num_classes)
    
    def forward(self, x):
        return self.base_model(x)


class EfficientNetClassifier(nn.Module):
    def __init__(self, num_classes=9):
        super(EfficientNetClassifier, self).__init__()

        self.base_model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

        self.base_model.classifier[1] = nn.Linear(
            self.base_model.classifier[1].in_features, num_classes
        )
    
    def forward(self, x):
        return self.base_model(x)


if __name__ == "__main__":
    ResNet = ResNetClassifier(num_classes=9)
    print(ResNet)
    
    EffNet = EfficientNetClassifier(num_classes=9)
    print(EffNet)