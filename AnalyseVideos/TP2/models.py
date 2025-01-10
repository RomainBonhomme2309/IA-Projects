import torch
import torch.nn as nn
from torchvision import models

class ResNetClassifier(nn.Module):
    def __init__(self, num_classes):
        super(ResNetClassifier, self).__init__()
        self.base_model = models.resnet50(pretrained=True)
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, num_classes)
    
    def forward(self, x):
        return self.base_model(x)


class EfficientNetClassifier(nn.Module):
    def __init__(self, num_classes=9):
        super(EfficientNetClassifier, self).__init__()

        self.base_model = models.efficientnet_b0(pretrained=True)

        self.base_model.classifier[1] = nn.Linear(
            self.base_model.classifier[1].in_features, num_classes
        )
    
    def forward(self, x):
        return self.base_model(x)


class UNetClassifier(nn.Module):
    def __init__(self, in_channels=3, num_classes=9):
        super(UNetClassifier, self).__init__()
        
        # Encoder
        self.enc1 = self.double_conv(in_channels, 64)
        self.enc2 = self.double_conv(64, 128)
        self.enc3 = self.double_conv(128, 256)
        self.enc4 = self.double_conv(256, 512)
        
        # Bottleneck
        self.bottleneck = self.double_conv(512, 1024)
        
        # Decoder
        self.upconv4 = self.upconv(1024, 512)
        self.dec4 = self.double_conv(1024, 512)
        self.upconv3 = self.upconv(512, 256)
        self.dec3 = self.double_conv(512, 256)
        self.upconv2 = self.upconv(256, 128)
        self.dec2 = self.double_conv(256, 128)
        self.upconv1 = self.upconv(128, 64)
        self.dec1 = self.double_conv(128, 64)
        
        # Final output layer for feature map
        self.final_conv = nn.Conv2d(64, 64, kernel_size=1)  # Reduce to 64 channels

        # Classification head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), # Global average pooling
            nn.Flatten(),                 # Flatten to 1D
            nn.Linear(64, num_classes)    # Fully connected layer for classification
        )
    
    def double_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def upconv(self, in_channels, out_channels):
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
    
    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(nn.MaxPool2d(2)(enc1))
        enc3 = self.enc3(nn.MaxPool2d(2)(enc2))
        enc4 = self.enc4(nn.MaxPool2d(2)(enc3))
        
        # Bottleneck
        bottleneck = self.bottleneck(nn.MaxPool2d(2)(enc4))
        
        # Decoder
        up4 = self.upconv4(bottleneck)
        dec4 = self.dec4(torch.cat([up4, enc4], dim=1))
        up3 = self.upconv3(dec4)
        dec3 = self.dec3(torch.cat([up3, enc3], dim=1))
        up2 = self.upconv2(dec3)
        dec2 = self.dec2(torch.cat([up2, enc2], dim=1))
        up1 = self.upconv1(dec2)
        dec1 = self.dec1(torch.cat([up1, enc1], dim=1))
        
        # Final feature map and classification
        features = self.final_conv(dec1)
        out = self.classifier(features)
        return out

if __name__ == "__main__":
    ResNet = ResNetClassifier(num_classes=9)
    print(ResNet)
    
    EffNet = EfficientNetClassifier(num_classes=9)
    print(EffNet)

    UNet = UNetClassifier(in_channels=3, num_classes=9)
    print(UNet)
