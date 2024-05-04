import torch
import torch.nn as nn
import torch.nn.functional as F


class FCN(nn.Module):
    def __init__(self, num_classes: int = 2, dropout: float = 0.3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            # nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Conv2d(32, 64, kernel_size=2, stride=1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            # nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Conv2d(64, 64, kernel_size=2, stride=1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
            # nn.MaxPool2d(kernel_size=2, stride=1),
            # nn.Conv2d(128, 64, kernel_size=1, stride=1),
            # nn.ReLU(inplace=True),
            # nn.Dropout(dropout),
            # nn.Conv2d(128, 128, kernel_size=2, stride=1),
            # nn.ReLU(inplace=True),
            # nn.Dropout(dropout),
            # nn.MaxPool2d(kernel_size=2, stride=1),
        )
        # self.avgpool = nn.AvgPool2d((2, 2))
        self.classifier = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=1),
                                        nn.Dropout(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        # x = self.avgpool(x)
        x = self.classifier(x)
        x = F.max_pool2d(x, kernel_size=x.size()[2:])
        sigmoid = nn.Sigmoid()
        output = sigmoid(x)
        return output

