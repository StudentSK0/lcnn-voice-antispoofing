import torch


def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """

    result_batch = {}

    # example of collate_fn
    result_batch["img"] = torch.cat(
        [elem["img"].unsqueeze(0) for elem in dataset_items], dim=0
    )
    result_batch["labels"] = torch.tensor([elem["labels"] for elem in dataset_items])

    return result_batch


def asvspoof_collate_fn(dataset_items: list[dict]) -> dict:
    """Combine fixed-length ASVspoof dataset items into a batch."""
    if not dataset_items:
        raise ValueError("dataset_items must contain at least one item")

    waveform = torch.stack([item["waveform"] for item in dataset_items])

    labels = torch.tensor(
        [item["labels"] for item in dataset_items],
        dtype=torch.long,
    )

    return {
        "waveform": waveform,
        "labels": labels,
        "audio_id": [item["audio_id"] for item in dataset_items],
        "speaker_id": [item["speaker_id"] for item in dataset_items],
        "attack_id": [item["attack_id"] for item in dataset_items],
    }
