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


class LCNNBlock3(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.conv_11 = nn.Conv2d(
            in_channels=48,
            out_channels=96,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

        self.mfm_12 = MaxFeatureMap()
        self.batch_norm_13 = nn.BatchNorm2d(48)

        self.conv_14 = nn.Conv2d(
            in_channels=48,
            out_channels=128,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )

        self.mfm_15 = MaxFeatureMap()
        self.pool_16 = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv_11(x)
        x = self.mfm_12(x)
        x = self.batch_norm_13(x)

        x = self.conv_14(x)
        x = self.mfm_15(x)
        x = self.pool_16(x)

        return x


class LCNNBlock4(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.conv_17 = nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

        self.mfm_18 = MaxFeatureMap()
        self.batch_norm_19 = nn.BatchNorm2d(64)

        self.conv_20 = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )

        self.mfm_21 = MaxFeatureMap()
        self.batch_norm_22 = nn.BatchNorm2d(32)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv_17(x)
        x = self.mfm_18(x)
        x = self.batch_norm_19(x)

        x = self.conv_20(x)
        x = self.mfm_21(x)
        x = self.batch_norm_22(x)

        return x


class LCNNBlock5(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.conv_23 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

        self.mfm_24 = MaxFeatureMap()
        self.batch_norm_25 = nn.BatchNorm2d(32)

        self.conv_26 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )

        self.mfm_27 = MaxFeatureMap()
        self.pool_28 = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv_23(x)
        x = self.mfm_24(x)
        x = self.batch_norm_25(x)

        x = self.conv_26(x)
        x = self.mfm_27(x)
        x = self.pool_28(x)

        return x


class LCNNFeatureExtractor(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.stem = LCNNStem()
        self.block_2 = LCNNBlock2()
        self.block_3 = LCNNBlock3()
        self.block_4 = LCNNBlock4()
        self.block_5 = LCNNBlock5()

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)
        x = self.block_5(x)

        return x


class LCNNClassifier(nn.Module):
    def __init__(self, dropout_probability: float = 0.75, num_classes: int = 2) -> None:
        super().__init__()

        self.flattened_features = 32 * 53 * 37

        self.flatten = nn.Flatten(start_dim=1)

        self.linear_29 = nn.Linear(
            in_features=self.flattened_features,
            out_features=160,
            bias=True,
        )

        self.mfm_30 = MaxFeatureMap()

        self.dropout = nn.Dropout(p=dropout_probability)
        self.batch_norm_31 = nn.BatchNorm1d(80)

        self.linear_32 = nn.Linear(
            in_features=80,
            out_features=num_classes,
            bias=True,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.flatten(x)
        x = self.linear_29(x)
        x = self.mfm_30(x)
        x = self.dropout(x)
        x = self.batch_norm_31(x)
        x = self.linear_32(x)

        return x


class LCNN(nn.Module):
    def __init__(self, dropout_probability: float = 0.75, num_classes: int = 2) -> None:
        super().__init__()

        self.feature_extractor = LCNNFeatureExtractor()
        self.classifier = LCNNClassifier(dropout_probability, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        features = self.feature_extractor(x)
        logits = self.classifier(features)

        return logits
