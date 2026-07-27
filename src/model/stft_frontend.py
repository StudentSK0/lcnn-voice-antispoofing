import torch
from torch import Tensor, nn


class LogPowerSTFT(nn.Module):
    def __init__(
        self,
        n_fft: int = 1724,
        win_length: int = 1724,
        hop_length: int = 130,
        n_frames: int = 600,
        eps: float = 1e-12,
    ) -> None:
        super().__init__()

        if n_fft <= 0:
            raise ValueError("n_fft must be greater than zero")

        if win_length <= 0:
            raise ValueError("win_length must be greater than zero")

        if win_length > n_fft:
            raise ValueError("win_length must not exceed n_fft")

        if hop_length <= 0:
            raise ValueError("hop_length must be greater than zero")

        if n_frames <= 0:
            raise ValueError("n_frames must be greater than zero")

        if eps <= 0:
            raise ValueError("eps must be greater than zero")

        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_frames = n_frames
        self.eps = eps

        self.target_length = n_fft + hop_length * (n_frames - 1)

        blackman_window = torch.blackman_window(win_length, periodic=True)
        self.register_buffer("blackman_window", blackman_window)

    def forward(self, waveform: Tensor) -> Tensor:
        expected_shape = f"[B, 1, {self.target_length}]"

        if (
            waveform.ndim != 3
            or waveform.shape[1] != 1
            or waveform.shape[-1] != self.target_length
        ):
            raise ValueError(
                f"Expected waveform shape {expected_shape}, "
                f"got {list(waveform.shape)}"
            )

        waveform = waveform.squeeze(1)

        spectrum = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.blackman_window,
            center=False,
            onesided=True,
            return_complex=True,
        )

        power = spectrum.abs().square()
        log_power = torch.log(power.clamp_min(self.eps))

        return log_power.unsqueeze(1)
