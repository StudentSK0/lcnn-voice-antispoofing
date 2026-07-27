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
