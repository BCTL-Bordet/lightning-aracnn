from torch import nn
import torchvision


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
