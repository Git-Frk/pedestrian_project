import torch
import torch.nn as nn
from torchvision import models
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.hub import load_state_dict_from_url
import torch.nn.functional as F

resnet18_url = 'https://download.pytorch.org/models/resnet18-5c106cde.pth'


# Defining a Fully Convolutional ResNet18 model:
# ---------------------------------------------
class FCResNet18(models.ResNet):
    def __init__(self, num_classes=2, pretrained=False, **kwargs):
        super().__init__(block=models.resnet.BasicBlock, layers=[2, 2, 2, 2], num_classes=num_classes, **kwargs)
        if pretrained:
            state_dict = load_state_dict_from_url(resnet18_url, progress=True)
            self.load_state_dict(state_dict)

        # Replace AdaptiveAvgPool2d with standard AvgPool2d
        self.avgpool = nn.AvgPool2d((2, 2))

        # Convert the original fc layer to a convolutional layer.
        # self.last_conv = torch.nn.Conv2d(in_channels=self.fc.in_features, out_channels=num_classes, kernel_size=1)
        # self.last_conv.weight.data.copy_(self.fc.weight.data.view(*self.fc.weight.data.shape, 1, 1))
        # self.last_conv.bias.data.copy_(self.fc.bias.data)
        self.classifier = nn.Sequential(nn.Conv2d(in_channels=self.fc.in_features, out_channels=num_classes, kernel_size=1),
                                        nn.Dropout(0.3))

    # Reimplementing forward pass.
    def _forward_impl(self, x):
        # Standard forward for resnet18
        x = self.conv1(x)
        # x = self.bn1(x)
        x = self.relu(x)
        # x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # x = self.avgpool(x)

        # Notice, there is no forward pass
        # through the original fully connected layer.
        # Instead, we forward pass through the last conv layer
        x = self.classifier(x)
        x = F.max_pool2d(x, kernel_size=x.size()[2:])
        return x


model = FCResNet18(num_classes=2)
# model.to(device)
