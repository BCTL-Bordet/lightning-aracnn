from torch import nn

from lightning_aracnn.model.components.blocks import ConvBlock, Head, ResidualBlock


class ARACNN(nn.Module):
    def __init__(
        self,
        num_classes: int,
        head_hidden_dim: int = 32,  # formerly "dropout_layer"
        dropout_rate: float = 0.5,
        num_blocks_first_path: int = 4,  # formerly "nb_of_residual_blocks_in_first_path"
        num_blocks_second_path: int = 3,  # formerlt "nb_of_residual_blocks_in_second_path"
    ):
        super().__init__()

        self.stem = ConvBlock(
            in_channels=3,
            out_channels=64,
            kernel_size=7,
            stride=4,
            padding=2,
            pooling_size=2,
        )

        self.residual_block_1 = nn.ModuleList(
            [
                ResidualBlock(
                    in_channels=64,
                    out_channels=64,
                    pooling=False,
                )
                for _ in range(num_blocks_first_path)
            ]
        )

        self.avg_pool_1 = nn.AvgPool2d(2)
        self.aux_head = Head(
            in_features=64,
            hidden_dim=head_hidden_dim,
            num_classes=num_classes,
            negative_slope=0.1,
            dropout_rate=dropout_rate,
        )
        self.residual_block_2 = nn.ModuleList(
            [
                ResidualBlock(
                    in_channels=64,
                    out_channels=64,
                    pooling=False,
                )
                for _ in range(num_blocks_second_path)
            ]
        )

        self.avg_pool_2 = nn.AvgPool2d(2)
        self.main_head = Head(
            in_features=64,
            hidden_dim=head_hidden_dim,
            num_classes=num_classes,
            negative_slope=0.1,
            dropout_rate=dropout_rate,
        )

    def forward(self, x):
        x = self.stem(x)
        for layer in self.residual_block_1:
            x = layer(x)
        x = self.avg_pool_1(x)

        x_aux = self.aux_head(x)
        for layer in self.residual_block_2:
            x = layer(x)

        x = self.avg_pool_2(x)

        x = self.main_head(x)
        return x, x_aux
