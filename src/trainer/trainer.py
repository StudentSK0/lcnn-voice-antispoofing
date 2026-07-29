import torch

from calculate_eer import compute_eer
from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer


class Trainer(BaseTrainer):
    """
    Trainer class. Defines the logic of batch logging and processing.
    """

    def process_batch(self, batch, metrics: MetricTracker):
        """
        Run batch through the model, compute metrics, compute loss,
        and do training step (during training stage).

        The function expects that criterion aggregates all losses
        (if there are many) into a single one defined in the 'loss' key.

        Args:
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics. The metrics depend on the type of
                the partition (train or inference).
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform),
                model outputs, and losses.
        """
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)  # transform batch on device -- faster

        metric_funcs = self.metrics["inference"]
        if self.is_train:
            metric_funcs = self.metrics["train"]
            self.optimizer.zero_grad()

        outputs = self.model(**batch)
        batch.update(outputs)

        if not self.is_train and self.config.trainer.get("compute_eer", False):
            scores = batch["logits"][:, 1] - batch["logits"][:, 0]
            self._evaluation_scores.append(scores.detach().cpu())
            self._evaluation_labels.append(batch["labels"].detach().cpu())

        all_losses = self.criterion(**batch)
        batch.update(all_losses)

        if self.is_train:
            batch["loss"].backward()  # sum of all losses is always called loss
            self._clip_grad_norm()
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        # update metrics for each loss (in case of multiple losses)
        for loss_name in self.config.writer.loss_names:
            metrics.update(loss_name, batch[loss_name].item())

        for met in metric_funcs:
            metrics.update(met.name, met(**batch))
        return batch

    def _evaluation_epoch(self, epoch, part, dataloader):
        """Evaluate one split and compute EER over all of its samples."""
        if not self.config.trainer.get("compute_eer", False):
            return super()._evaluation_epoch(epoch, part, dataloader)

        self._evaluation_scores = []
        self._evaluation_labels = []

        logs = super()._evaluation_epoch(epoch, part, dataloader)

        scores = torch.cat(self._evaluation_scores)
        labels = torch.cat(self._evaluation_labels)

        if not torch.isfinite(scores).all():
            raise ValueError(f"Non-finite scores encountered on the {part} split")

        bonafide_scores = scores[labels == 1].numpy()
        spoof_scores = scores[labels == 0].numpy()

        if bonafide_scores.size == 0 or spoof_scores.size == 0:
            raise ValueError(
                f"EER requires both bonafide and spoof samples on the {part} split"
            )

        eer, _ = compute_eer(bonafide_scores, spoof_scores)
        eer_percent = float(eer) * 100.0
        logs["EER"] = eer_percent

        self.writer.add_scalar("EER", eer_percent)

        return logs

    def _log_batch(self, batch_idx, batch, mode="train"):
        """
        Batch-level media logging hook.

        Args:
            batch_idx (int): index of the current batch.
            batch (dict): dict-based batch after going through
                the 'process_batch' function.
            mode (str): train or inference. Defines which logging
                rules to apply.
        """
        pass
