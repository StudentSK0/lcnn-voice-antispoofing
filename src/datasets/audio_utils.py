import torch
import torch.nn.functional as F
from torch import Tensor


def fix_audio_length(
    waveform: Tensor,
    target_length: int,
    random_crop: bool,
) -> Tensor:
    if waveform.ndim != 2:
        raise ValueError("waveform must have shape [channels, samples]")

    if waveform.shape[0] == 0:
        raise ValueError("waveform must contain at least one channel")

    if target_length <= 0:
        raise ValueError("target_length must be greater than zero")

    current_length = waveform.shape[-1]

    if current_length == 0:
        raise ValueError("waveform must contain at least one sample")

    if current_length == target_length:
        return waveform

    if current_length < target_length:
        padding_length = target_length - current_length
        return F.pad(waveform, (0, padding_length))

    max_start = current_length - target_length

    if random_crop:
        start = torch.randint(
            low=0,
            high=max_start + 1,
            size=(1,),
            device=waveform.device,
        ).item()
    else:
        start = 0

    end = start + target_length
    return waveform[..., start:end]
