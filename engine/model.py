import torch.nn.functional as F
from torch import nn
from torchvision.models import mobilenet_v3_small
from torchvision.models import MobileNet_V3_Small_Weights

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.linear_out = nn.Linear(576, 4)

    def forward(self, x):
        x = self.linear_out(x)
        x = F.sigmoid(x)
        return x

class SpaceshipDetector(nn.Module):
    def __init__(self):
        super(SpaceshipDetector, self).__init__()
        self.detector = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        self.detector.classifier = Classifier()

    def get_layer_num(self):
        return len(self.detector.features)

    def forward(self, x):
        x = self.detector(x)
        return x
