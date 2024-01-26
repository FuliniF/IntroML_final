from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

base_model = resnet50(weights=ResNet50_Weights.DEFAULT).to(device)
for param in base_model.parameters():
    param.requires_grad = True

class Resnet50Model(nn.Module):
    def __init__(self, num_classes=200):
        super(Resnet50Model, self).__init__()
        self.resnet = base_model
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.resnet(x)

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.load_state_dict(checkpoint)
        print(f"Model loaded from {checkpoint_path}")

    def forward(self, x):
        return self.resnet(x)

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.load_state_dict(checkpoint)
        print(f"Model loaded from {checkpoint_path}")