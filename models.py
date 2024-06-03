import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class L1Dist(nn.Module):
    def __init__(self):
        super(L1Dist, self).__init__()

    def forward(self, x, y):
        return torch.abs(x - y)
    

class ResNetEncoder(nn.Module):
    def __init__(self):
        super(ResNetEncoder, self).__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(resnet.fc.in_features, 4096)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)  # Flatten the output
        x = self.fc(x)
        return x

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.embedding = ResNetEncoder()
        self.l1_dist = L1Dist()
        self.classifier = nn.Sequential(
            nn.Linear(4096, 1),
            nn.Sigmoid()
        )

    def forward(self, input_img, validation_img):
        input_embedding = self.embedding(input_img)
        validation_embedding = self.embedding(validation_img)
        distance = self.l1_dist(input_embedding, validation_embedding)
        output = self.classifier(distance)
        return output
