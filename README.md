# LCNN Voice Anti-Spoofing

LCNN countermeasure for classifying *bonafide* and *spoof* speech in the
ASVspoof 2019 LA. The pipeline combines a log-power STFT front-end,
Max-Feature-Map activations and cross-entropy loss.

> [!NOTE]
> The submitted epoch 9 checkpoint achieved **7.9122% EER** on the LA evaluation set.

## How to use

Set `ASVSPOOF_ROOT` to the LA dataset directory, authenticate with W&B, and
start training:

```bash
export ASVSPOOF_ROOT=/absolute/path/to/LA/LA
wandb login

python train.py -cn=asvspoof \
  trainer.n_epochs=10 \
  trainer.save_period=1 \
  writer.run_name=lcnn-ce-seed1-restart
```

Generate the submission from the selected checkpoint:

```bash
ASVSPOOF_CHECKPOINT=saved/lcnn-ce-seed1-restart/checkpoint-epoch9.pth \
SUBMISSION_FILENAME=visekorobov.csv \
python inference.py -cn=asvspoof_inference
```

## Credits

This repository is based on the
[PyTorch Project Template](https://github.com/Blinorot/pytorch_project_template)
provided for the course. The LCNN training recipe follows the
[STC ASVspoof 2019 paper](https://arxiv.org/abs/1904.05576), and EER is computed
with the course-provided `calculate_eer.py`.

## License

This project is released under the [MIT License](LICENSE).
