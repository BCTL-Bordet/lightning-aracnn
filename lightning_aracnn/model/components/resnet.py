from torch import nn
import torchvision
from collections import OrderedDict
import torch


class ResNet(nn.Module):
    def __init__(
        self,
        resnet: torchvision.models.resnet.ResNet,
        num_classes: int = 10,
        freeze: bool = True,
    ):
        super().__init__()
        self.net = resnet

        # switch out last fully connected layer to fit our number of classes
        self.net.fc = nn.Linear(
            in_features=self.net.fc.in_features,
            out_features=num_classes,
        )

        # add Softmax as last layer
        self.net = nn.Sequential(
            self.net,
            nn.Softmax(),
        )

        # stop gradients for parameters in convolutional layers
        if freeze:
            self.freeze_backbone()

    def freeze_backbone(self):
        for name, param in self.net.named_parameters():
            if "fc" not in name:
                param.requires_grad = False

    def forward(self, x):
        return self.net(x)


class MultiLabelResNet(nn.Module):
    def __init__(
        self,
        resnet: torchvision.models.resnet.ResNet,
        num_classes: int = 10,
        freeze: bool = True,
    ):
        super().__init__()
        self.backbone = resnet
        self.in_features = self.backbone.fc.in_features

        # remove classifier from network
        self.backbone = nn.Sequential(
            OrderedDict(list(self.backbone.named_children())[:-1])
        )

        # create one classifier head for each class
        self.heads = nn.ModuleList(
            [nn.Linear(self.in_features, 1) for _ in range(num_classes)]
        )

        # stop gradients for parameters in convolutional layers
        if freeze:
            self.freeze_backbone()

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, x):
        """
        Features go through each classifier head to perform binary classification against all labels
        """
        features = self.backbone(x)
        features = torch.flatten(features, 1)
        logits = [head(features) for head in self.heads]

        return logits
