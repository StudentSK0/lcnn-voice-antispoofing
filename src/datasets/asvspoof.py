from pathlib import Path
from typing import TypedDict

import torchaudio
from torch.utils.data import Dataset

from src.datasets.audio_utils import fix_audio_length


class ASVspoofIndexEntry(TypedDict):
    path: str
    speaker_id: str
    audio_id: str
    attack_id: str
    label: int


class ProtocolEntry(TypedDict):
    speaker_id: str
    audio_id: str
    attack_id: str
    label: int


class ASVspoofDataset(Dataset):
    def __init__(
        self,
        protocol_path: str | Path,
        audio_dir: str | Path,
        target_length: int,
        expected_sample_rate: int = 16000,
        random_crop: bool = False,
    ) -> None:
        if target_length <= 0:
            raise ValueError(
                f"target_length must be greater than zero, got {target_length}"
            )

        if expected_sample_rate <= 0:
            raise ValueError(
                "expected_sample_rate must be greater than zero, "
                f"got {expected_sample_rate}"
            )

        self.index = build_asvspoof_index(
            protocol_path=protocol_path,
            audio_dir=audio_dir,
        )

        self.target_length = target_length
        self.expected_sample_rate = expected_sample_rate
        self.random_crop = random_crop

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, index: int) -> dict:
        entry = self.index[index]
        audio_path = entry["path"]

        waveform, sample_rate = torchaudio.load(audio_path)

        if sample_rate != self.expected_sample_rate:
            raise ValueError(
                f"Expected sample rate {self.expected_sample_rate} Hz, "
                f"got {sample_rate} Hz for file {audio_path}"
            )

        if waveform.ndim != 2:
            raise ValueError(
                "Expected waveform shape [channels, samples], "
                f"got {list(waveform.shape)} for file {audio_path}"
            )

        if waveform.shape[0] != 1:
            raise ValueError(
                "Expected mono waveform with shape [1, samples], "
                f"got {list(waveform.shape)} for file {audio_path}"
            )

        waveform = fix_audio_length(
            waveform=waveform,
            target_length=self.target_length,
            random_crop=self.random_crop,
        )

        return {
            "waveform": waveform,
            "labels": entry["label"],
            "audio_id": entry["audio_id"],
            "speaker_id": entry["speaker_id"],
            "attack_id": entry["attack_id"],
        }


def parse_protocol_line(line: str) -> ProtocolEntry:
    fields = line.strip().split()

    if len(fields) == 5:
        protocol = ProtocolEntry()

        protocol["speaker_id"] = fields[0]
        protocol["audio_id"] = fields[1]
        protocol["attack_id"] = fields[3]
        if fields[4] == "spoof":
            protocol["label"] = 0
        elif fields[4] == "bonafide":
            protocol["label"] = 1
        else:
            raise ValueError(f"invalid label: {fields[4]}")
    else:
        raise ValueError(f"invalid protocol line: {line}")
    return protocol


def read_protocol(protocol_path: str | Path) -> list[ProtocolEntry]:
    protocol_path = Path(protocol_path)
    protocol_entries: list[ProtocolEntry] = []

    with protocol_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                entry = parse_protocol_line(line)
            except ValueError as error:
                raise ValueError(
                    f"invalid entry at line {line_number} "
                    + f"in {protocol_path}: {error}"
                ) from error

            protocol_entries.append(entry)

    return protocol_entries


def validate_unique_audio_ids(entries: list[ProtocolEntry]) -> None:
    seen = set()
    for entry in entries:
        audio_id = entry["audio_id"]
        if audio_id in seen:
            raise ValueError(f"Duplicate audio_id found: {audio_id}")
        seen.add(audio_id)


def build_asvspoof_index(
    protocol_path: str | Path, audio_dir: str | Path
) -> list[ASVspoofIndexEntry]:
    protocol_entries = read_protocol(protocol_path)
    validate_unique_audio_ids(protocol_entries)
    audio_dir = Path(audio_dir)

    index_entries = []
    for entry in protocol_entries:
        audio_path = audio_dir / f"{entry['audio_id']}.flac"
        if not audio_path.is_file():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        index_entry = ASVspoofIndexEntry(
            path=str(audio_path),
            speaker_id=entry["speaker_id"],
            audio_id=entry["audio_id"],
            attack_id=entry["attack_id"],
            label=entry["label"],
        )
        index_entries.append(index_entry)

    return index_entries


if __name__ == "__main__":
    print(parse_protocol_line("LA_0039 LA_E_2834763 - A11 spoof"))
    print(parse_protocol_line("LA_0069 LA_D_1047731 - - bonafide"))
