from pathlib import Path

from lightning import Callback, Trainer
from dataloader import load_and_create_dataloaders
import torch

# import lightning as pl
# import lightning
from lightning.pytorch.loggers import WandbLogger

# .pytorch.loggers.wandb
import wandb
from dataclasses import dataclass, asdict
from typing import Literal, Optional, Union
import random
from lightning.pytorch.callbacks import (
  LearningRateFinder,
  ModelCheckpoint,
  OnExceptionCheckpoint,
  RichProgressBar,
)
from eegpt import EegptLightning, EegptConfig


@dataclass
class TrainingConfig:
  eegpt_chpt_path: Path = Path(
    "./model_checkpoints/25866970/EEGPT/checkpoint/eegpt_mcae_58chs_4s_large4E.ckpt"
  )
  data_path: Path = Path("./datasets/bcmi_combined_prepared_mel_28ch")
  data_loader_num_workers: int = 4
  batch_size: int = 8
  lr: float = 1e-4
  num_epochs: int = 100
  save_model_per_epochs: int = 5
  val_every_n_epoch: int = 5
  ckpt_load_path: Optional[str] = None  # 'best', 'last', <path]>
  wandb_log_model: Union[Literal["all"], bool] = "all"
  project_name: str = "neural-music-decoding"
  run_name: str = "eegpt-2layer-mel"
  run_extra_name: str = "lr_find"
  randint: int = random.randint(0, 1000)
  save_path: str = f"{run_name}-ckpt"
  use_learning_rate_finder: bool = False


config = TrainingConfig()


def count_n_params(model):
  """Counts the number of trainable parameters in a model."""
  return sum(p.numel() for p in model.parameters() if p.requires_grad)


def log_spectrograms(pl_module, y_hat, y, batch_idx, stage: str, n_samples=4):
  """Log a batch of predicted and ground truth spectrograms to wandb."""
  y_hat = y_hat.detach().cpu()[:n_samples]
  y = y.detach().cpu()[:n_samples]

  images = []
  for i, (pred_spec, true_spec) in enumerate(zip(y_hat, y)):
    # # Normalize for visualization
    # pred_spec = (pred_spec - pred_spec.min()) / (pred_spec.max() - pred_spec.min() + 1e-8)
    # true_spec = (true_spec - true_spec.min()) / (true_spec.max() - true_spec.min() + 1e-8)

    # Combine pred and true for side-by-side comparison
    combined_spec = torch.cat((pred_spec, true_spec), dim=1)

    images.append(
      wandb.Image(combined_spec.numpy(), caption=f"Pred vs. True (Sample {i})")
    )

  pl_module.logger.experiment.log(
    {f"{stage}/spectrograms": images}, step=pl_module.global_step
  )


class SpectrogramLoggingCallback(Callback):
  def __init__(self):
    super().__init__()
    self.val_log_batch_idx = 0
    self.test_log_batch_idx = 0

  def on_validation_epoch_start(self, trainer, pl_module):
    """Choose a random batch to log for this validation epoch."""
    if trainer.val_dataloaders:
      num_batches = len(trainer.val_dataloaders)
      if num_batches > 0:
        self.val_log_batch_idx = random.randint(0, num_batches - 1)

  def on_test_epoch_start(self, trainer, pl_module):
    """Choose a random batch to log for this test epoch."""
    if trainer.test_dataloaders:
      # Assuming single test dataloader
      num_batches = len(trainer.test_dataloaders)
      if num_batches > 0:
        self.test_log_batch_idx = random.randint(0, num_batches - 1)

  def on_validation_batch_end(
    self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
  ):
    """Log spectrograms at the end of each validation batch."""
    if batch_idx == self.val_log_batch_idx:
      x = batch["eeg"]
      y = batch["mel"]
      y_hat = pl_module(x)
      log_spectrograms(pl_module, y_hat, y, batch_idx, stage="val")

  def on_test_batch_end(
    self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
  ):
    """Log spectrograms at the end of each test batch."""
    if batch_idx == self.test_log_batch_idx:
      x = batch["eeg"]
      y = batch["mel"]
      y_hat = pl_module(x)
      log_spectrograms(pl_module, y_hat, y, batch_idx, stage="test")


def main(config=config):
  device = "cuda" if torch.cuda.is_available() else "cpu"
  dataloaders = load_and_create_dataloaders(config.data_path, config)

  eegpt_config = EegptConfig(chpt_path=config.eegpt_chpt_path, lr=config.lr)
  model = EegptLightning(eegpt_config)

  wandb_logger = WandbLogger(
    project=config.project_name,
    name=f"{config.run_name}-{config.run_extra_name}-{config.randint}",
    log_model=config.wandb_log_model,
    config=asdict(config),
  )

  wandb_logger.watch(model, log="all")

  save_on_exc = OnExceptionCheckpoint(
    f"{config.save_path}/exc_save",
  )

  ckpt_callback = ModelCheckpoint(
    every_n_epochs=config.save_model_per_epochs,
    dirpath=config.save_path,
    save_top_k=2,
    monitor="val_loss",
    mode="min",
    save_last=True,
  )

  optional_lr_finder = (
    [LearningRateFinder(min_lr=1e-08, max_lr=1, num_training_steps=100)]
    if config.use_learning_rate_finder
    else []
  )

  trainer = Trainer(
    callbacks=[
      ckpt_callback,
      SpectrogramLoggingCallback(),
      RichProgressBar(),
      save_on_exc,
    ]
    + optional_lr_finder,
    logger=wandb_logger,
    check_val_every_n_epoch=config.val_every_n_epoch,
    max_epochs=config.num_epochs,
    accelerator=device,
  )

  print(f"Model trainable params: {count_n_params(model)}")

  trainer.fit(
    model,
    train_dataloaders=dataloaders["train"],
    val_dataloaders=dataloaders["val"],
    ckpt_path=config.ckpt_load_path,
  )

  trainer.test(
    model,
    dataloaders=dataloaders["test"],
  )


if __name__ == "__main__":
  main()
