from torch import Tensor, nn

from src.model.mfm import MaxFeatureMap


class LCNNStem(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=5,
            stride=1,
            padding=2,
            bias=True,
        )
        self.mfm = MaxFeatureMap()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.mfm(x)
        x = self.pool(x)
        return x


class LCNNBlock2(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.conv_4 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )
        self.mfm_5 = MaxFeatureMap()
        self.batch_norm_6 = nn.BatchNorm2d(32)

        self.conv_7 = nn.Conv2d(
            in_channels=32,
            out_channels=96,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )
        self.mfm_8 = MaxFeatureMap()
        self.pool_9 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.batch_norm_10 = nn.BatchNorm2d(48)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv_4(x)
        x = self.mfm_5(x)
        x = self.batch_norm_6(x)

        x = self.conv_7(x)
        x = self.mfm_8(x)
        x = self.pool_9(x)
        x = self.batch_norm_10(x)

        return x
