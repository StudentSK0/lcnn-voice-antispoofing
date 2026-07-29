from torch import Tensor, nn

from src.model.lcnn import LCNN
from src.model.stft_frontend import LogPowerSTFT


class VoiceAntiSpoofingModel(nn.Module):
    def __init__(
        self,
        n_fft: int = 1724,
        win_length: int = 1724,
        hop_length: int = 130,
        n_frames: int = 600,
        stft_eps: float = 1e-12,
        dropout_probability: float = 0.75,
        num_classes: int = 2,
    ) -> None:
        super().__init__()

        self.frontend = LogPowerSTFT(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_frames=n_frames,
            eps=stft_eps,
        )

        self.lcnn = LCNN(
            dropout_probability=dropout_probability, num_classes=num_classes
        )

    def forward(self, waveform: Tensor, **_: object) -> dict[str, Tensor]:
        spectrogram = self.frontend(waveform)
        logits = self.lcnn(spectrogram)

        return {"logits": logits}
