import csv
from pathlib import Path

import torch

from calculate_eer import compute_eer
from src.trainer.inferencer import Inferencer


class ASVspoofInferencer(Inferencer):
    """Run ASVspoof inference and save bonafide scores to one CSV file."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        output_filename = self.config.inferencer.output_filename
        if Path(output_filename).name != output_filename:
            raise ValueError("output_filename must not contain directories")
        if not output_filename.endswith(".csv"):
            raise ValueError("output_filename must end with .csv")
        if self.save_path is None:
            raise ValueError("save_path is required for ASVspoof inference")

        self.output_filename = output_filename
        self._prediction_rows: dict[str, list[tuple[str, float]]] = {}
        self._labels: dict[str, list[int]] = {}

    def process_batch(self, batch_idx, batch, metrics, part):
        """Calculate scores for one batch and retain them for CSV output."""
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)

        outputs = self.model(**batch)
        batch.update(outputs)

        logits = batch["logits"]
        if logits.ndim != 2 or logits.shape[1] != 2:
            raise ValueError(f"Expected logits shape [B, 2], got {list(logits.shape)}")

        scores = (logits[:, 1] - logits[:, 0]).detach().cpu()
        if not torch.isfinite(scores).all():
            raise ValueError(f"Non-finite scores encountered on the {part} split")

        audio_ids = batch["audio_id"]
        if len(audio_ids) != scores.shape[0]:
            raise ValueError("audio_id and score batch sizes do not match")

        labels = batch["labels"].detach().cpu()
        if labels.ndim != 1 or labels.shape[0] != scores.shape[0]:
            raise ValueError("label and score batch sizes do not match")

        self._prediction_rows[part].extend(
            (audio_id, float(score))
            for audio_id, score in zip(audio_ids, scores.tolist())
        )
        self._labels[part].extend(int(label) for label in labels.tolist())

        if metrics is not None:
            for metric in self.metrics["inference"]:
                metrics.update(metric.name, metric(**batch))

        return batch

    def _inference_part(self, part, dataloader):
        self._prediction_rows[part] = []
        self._labels[part] = []
        logs = super()._inference_part(part, dataloader)

        rows = self._prediction_rows[part]
        labels = self._labels[part]
        dataset_size = len(dataloader.dataset)
        if len(rows) != dataset_size:
            raise ValueError(f"Expected {dataset_size} predictions, got {len(rows)}")
        if len(labels) != len(rows):
            raise ValueError("label and prediction counts do not match")

        audio_ids = [audio_id for audio_id, _ in rows]
        if len(set(audio_ids)) != len(audio_ids):
            raise ValueError(f"Duplicate audio_id values found on the {part} split")

        expected_num_predictions = self.config.inferencer.get(
            "expected_num_predictions"
        )
        if (
            expected_num_predictions is not None
            and len(rows) != expected_num_predictions
        ):
            raise ValueError(
                f"Expected {expected_num_predictions} predictions, got {len(rows)}"
            )

        output_path = self.save_path / part / self.output_filename
        with output_path.open("w", encoding="utf-8", newline="") as file:
            csv.writer(file).writerows(rows)

        scores = torch.tensor([score for _, score in rows])
        labels_tensor = torch.tensor(labels)
        bonafide_scores = scores[labels_tensor == 1].numpy()
        spoof_scores = scores[labels_tensor == 0].numpy()
        if bonafide_scores.size == 0 or spoof_scores.size == 0:
            raise ValueError(
                f"EER requires both bonafide and spoof samples on the {part} split"
            )

        eer, _ = compute_eer(bonafide_scores, spoof_scores)
        logs["EER"] = float(eer) * 100.0
        logs["num_predictions"] = len(rows)
        print(f"Predictions saved to {output_path}")
        return logs
