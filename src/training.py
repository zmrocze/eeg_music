from pathlib import Path

from lightning import Callback, Trainer
from eeg_music.dataloader import load_and_create_dataloaders
import torch
from skimage.metrics import structural_similarity
from sklearn.metrics.pairwise import cosine_similarity

# import lightning as pl
# import lightning
from lightning.pytorch.loggers import WandbLogger

# .pytorch.loggers.wandb
import wandb
from dataclasses import dataclass, asdict, field
from typing import Literal, Union
import random
from lightning.pytorch.callbacks import (
  LearningRateFinder,
  ModelCheckpoint,
  OnExceptionCheckpoint,
  RichProgressBar,
)
from eeg_music.eegpt import (
  EegptLightning,
  EegptConfig,
  LRCosine,
  EEG_WIDTH,
  USING_CHANNELS,
)
from eeg_music.freeze_utils import freeze_all_except_head_and_adapters


@dataclass
class TrainingConfig:
  eegpt_chpt_path: Path = Path(
    "./model_checkpoints/25866970/EEGPT/checkpoint/eegpt_mcae_58chs_4s_large4E.ckpt"
  )
  data_path: Path = Path("./datasets/bcmi_preprocessed/bcmi_combined_prepared_mel_28ch")
  data_loader_num_workers: int = 4
  prefetch_factor: int = 2
  batch_size: int = 8
  num_epochs: int = 100
  save_model_per_epochs: int = 5

  val_every_n_epoch: int = 1
  ds_p_train = 0.85
  ds_p_val = 0.0
  ds_split_seed = 42
  ds_use_test_for_val = True
  ds_test_repeated_mul = 10

  # ckpt_load_path: Optional[str] = None  # 'best', 'last', <path]>

  wandb_log_model: Union[Literal["all"], bool] = "all"
  project_name: str = "neural-music-decoding"
  run_name: str = "eegpt-2layer-mel"
  run_extra_name: str = "lr_find"
  randint: int = random.randint(0, 1000)
  save_path: str = f"{run_name}-ckpt"

  lr_config: Union[float, LRCosine] = 1e-4

  use_learning_rate_finder: bool = False

  # Freezing strategy
  freeze_layers: bool = (
    False  # If True, freeze all except chan_conv, head, and ResidualLinear
  )
  use_chan_conv: bool = True
  num_classes: int = 128

  # AUROC callback settings
  auroc_every_n_epochs: int = 2
  auroc_similarity_metric: list[Literal["cosine", "structural_similarity"]] = field(
    default_factory=lambda: ["cosine", "structural_similarity"]
  )


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


class AUROCCallback(Callback):
  """Calculate AUROC-like retrieval metric on validation set.

  For each EEG sample, compares predicted mel spectrogram to all validation
  mel spectrograms and computes the rank of the correct match.
  """

  def __init__(self, auroc_every_n_epochs: int = 5, similarity_metric: str = "cosine"):
    """
    Args:
      auroc_every_n_epochs: Calculate AUROC score every N epochs
      similarity_metric: Either 'cosine' or 'structural_similarity'
    """
    super().__init__()
    self.auroc_every_n_epochs = auroc_every_n_epochs
    self.similarity_metric = similarity_metric
    self.auroc_history = []  # Store last 10 scores for moving average
    # Create suffix for metric names to distinguish different similarity metrics
    self.metric_suffix = (
      "" if similarity_metric == "cosine" else f"_{similarity_metric}"
    )

  def on_validation_epoch_end(self, trainer, pl_module):
    """Calculate AUROC score at the end of validation epoch."""
    # Only calculate every N epochs
    if (trainer.current_epoch + 1) % self.auroc_every_n_epochs != 0:
      return

    # Check if validation dataloader exists
    if not trainer.val_dataloaders:
      return

    # Collect all validation data
    all_x = []
    all_y = []

    pl_module.eval()
    with torch.no_grad():
      for batch in trainer.val_dataloaders:
        x = batch["eeg"].to(pl_module.device)
        y = batch["mel"].to(pl_module.device)
        all_x.append(x)
        all_y.append(y)

    # Concatenate all batches
    all_x = torch.cat(all_x, dim=0)  # Shape: (N, channels, time)
    all_y = torch.cat(all_y, dim=0)  # Shape: (N, freq, time)

    # Generate predictions for all samples
    with torch.no_grad():
      all_y_hat = pl_module(all_x)  # Shape: (N, freq, time)

    n_samples = all_y.shape[0]
    ranks = []

    # For each sample, compute similarity to all targets
    for i in range(n_samples):
      y_hat_i = all_y_hat[i]  # Shape: (freq, time)
      similarities = []

      # Compare to all ground truth spectrograms
      for j in range(n_samples):
        y_j = all_y[j]  # Shape: (freq, time)
        sim = self._compute_similarity(y_hat_i, y_j)
        similarities.append(sim)

      # Sort similarities in descending order (higher similarity = better match)
      similarities = torch.tensor(similarities)
      sorted_indices = torch.argsort(similarities, descending=True)

      # Find rank of correct match (where sorted_indices == i)
      rank = (sorted_indices == i).nonzero(as_tuple=True)[0].item()
      ranks.append(rank)

    # Calculate AUROC-like score
    # If correct match is rank 0 (best), score should be 1.0
    # If correct match is rank (n_samples-1) (worst), score should be 0.0
    auroc_scores = [1.0 - (rank / (n_samples - 1)) for rank in ranks]
    mean_auroc = sum(auroc_scores) / len(auroc_scores)

    # Update history for moving average
    self.auroc_history.append(mean_auroc)
    if len(self.auroc_history) > 10:
      self.auroc_history.pop(0)

    moving_avg = sum(self.auroc_history) / len(self.auroc_history)

    # Log metrics
    pl_module.log(
      f"auroc_score{self.metric_suffix}",
      mean_auroc,
      on_epoch=True,
      prog_bar=True,
      logger=True,
    )
    pl_module.log(
      f"auroc_ma10{self.metric_suffix}",
      moving_avg,
      on_epoch=True,
      prog_bar=True,
      logger=True,
    )

    # Log distribution of ranks for debugging
    median_rank = sorted(ranks)[len(ranks) // 2]
    pl_module.log(
      f"auroc_median_rank{self.metric_suffix}", median_rank, on_epoch=True, logger=True
    )

    # Log top-k accuracy metrics
    pl_module.log(
      f"auroc_top1_accuracy{self.metric_suffix}",
      sum(1 for r in ranks if r == 0) / len(ranks),
      on_epoch=True,
      logger=True,
    )
    pl_module.log(
      f"auroc_top10_accuracy{self.metric_suffix}",
      sum(1 for r in ranks if r < 10) / len(ranks),
      on_epoch=True,
      logger=True,
    )
    pl_module.log(
      f"auroc_top25_accuracy{self.metric_suffix}",
      sum(1 for r in ranks if r < 25) / len(ranks),
      on_epoch=True,
      logger=True,
    )
    pl_module.log(
      f"auroc_top100_accuracy{self.metric_suffix}",
      sum(1 for r in ranks if r < 100) / len(ranks),
      on_epoch=True,
      logger=True,
    )

  def _compute_similarity(self, pred, target):
    """Compute similarity between two spectrograms.

    Args:
      pred: Predicted spectrogram (freq, time)
      target: Target spectrogram (freq, time)

    Returns:
      Similarity score (higher = more similar)
    """
    if self.similarity_metric == "cosine":
      # Flatten and compute cosine similarity
      pred_flat = pred.reshape(-1).cpu().numpy()
      target_flat = target.reshape(-1).cpu().numpy()
      # cosine_similarity expects 2D arrays
      sim = cosine_similarity(pred_flat.reshape(1, -1), target_flat.reshape(1, -1))[
        0, 0
      ]
      return sim

    elif self.similarity_metric == "structural_similarity":
      # Convert to numpy and compute SSIM
      pred_np = pred.cpu().numpy()
      target_np = target.cpu().numpy()
      # SSIM expects 2D grayscale images
      sim = structural_similarity(
        pred_np,
        target_np,
        data_range=max(
          pred_np.max() - pred_np.min(), target_np.max() - target_np.min()
        ),
      )
      return sim

    else:
      raise ValueError(f"Unknown similarity metric: {self.similarity_metric}")


def log_hyperparameters(model, dataloaders, config, wandb_logger):
  params_to_log = {}

  # Parameter counts
  params_to_log["trainable_params_total"] = count_n_params(model)
  if hasattr(model.model.model, "chan_conv"):
    params_to_log["trainable_params_chan_conv"] = count_n_params(
      model.model.model.chan_conv
    )
  params_to_log["trainable_params_residual_linear"] = count_n_params(model.model.linear)
  params_to_log["trainable_params_head"] = count_n_params(model.model.model.head)

  # ResidualLinear dimensions
  params_to_log["residual_linear_in_dim"] = model.model.linear.linear1.in_features
  params_to_log["residual_linear_out_dim"] = model.model.linear.linear2.out_features

  # EEGPTClassifier params
  eegpt_classifier = model.model.model
  params_to_log["eegpt_classifier_use_chan_conv"] = eegpt_classifier.use_chan_conv

  # Target Encoder params
  target_encoder = eegpt_classifier.target_encoder
  params_to_log["target_encoder_img_size"] = str(target_encoder.patch_embed.img_size)
  params_to_log["target_encoder_patch_size"] = target_encoder.patch_embed.patch_size
  params_to_log["target_encoder_embed_dim"] = target_encoder.embed_dim
  params_to_log["target_encoder_depth"] = len(target_encoder.blocks)
  params_to_log["target_encoder_num_heads"] = target_encoder.num_heads
  params_to_log["target_encoder_patch_stride"] = target_encoder.patch_embed.patch_stride

  # Predictor params
  if eegpt_classifier.use_predictor:
    predictor = eegpt_classifier.predictor
    params_to_log["predictor_embed_dim"] = predictor.predictor_embed.in_features
    params_to_log["predictor_depth"] = len(predictor.predictor_blocks)
    params_to_log["predictor_num_heads"] = predictor.predictor_blocks[0].attn.num_heads

  # EEG data params
  params_to_log["eeg_width"] = EEG_WIDTH
  params_to_log["using_channels"] = USING_CHANNELS

  # Dataloader params
  params_to_log["dataloader_train_size"] = len(dataloaders["train"])
  params_to_log["dataloader_val_size"] = len(dataloaders["val"])
  params_to_log["dataloader_test_size"] = len(dataloaders["test"])
  params_to_log["batch_size"] = config.batch_size
  params_to_log["num_workers"] = config.data_loader_num_workers

  wandb_logger.log_hyperparams(params_to_log)


def main(config=config):
  print("DOEs it even work???")
  dataloaders = load_and_create_dataloaders(config.data_path, config)
  assert (
    isinstance(config.lr_config, float) if config.use_learning_rate_finder else True
  )
  eegpt_config = EegptConfig(
    chpt_path=config.eegpt_chpt_path,
    lr_config=config.lr_config,
    num_classes=config.num_classes,
    use_chan_conv=config.use_chan_conv,
  )
  model = EegptLightning(eegpt_config)

  # Apply freezing strategy if enabled
  if config.freeze_layers:
    print("\nApplying layer freezing strategy...")
    freeze_all_except_head_and_adapters(model, verbose=True)

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

  # Create AUROC callbacks for each similarity metric
  auroc_callbacks = [
    AUROCCallback(
      auroc_every_n_epochs=config.auroc_every_n_epochs,
      similarity_metric=metric,
    )
    for metric in config.auroc_similarity_metric
  ]

  trainer = Trainer(
    callbacks=[
      ckpt_callback,
      SpectrogramLoggingCallback(),
      RichProgressBar(),
      save_on_exc,
    ]
    + auroc_callbacks
    + optional_lr_finder,
    logger=wandb_logger,
    check_val_every_n_epoch=config.val_every_n_epoch,
    max_epochs=config.num_epochs,
    accelerator="auto",
    precision="16-mixed"
  )

  print("trainer_precision: ", trainer.precision)
  print("trainer_precision: ", trainer.precision)
  print("trainer_precision: ", trainer.precision)

  print(f"Model trainable params: {count_n_params(model)}")
  print(
    "Note that val and test dataloaders augmentation/randomness in the form of choosing the 4s fragment."
  )

  log_hyperparameters(model, dataloaders, config, wandb_logger)

  print("trainer_precision: ", trainer.precision)

  trainer.fit(
    model,
    train_dataloaders=dataloaders["train"],
    val_dataloaders=dataloaders["val"],
    # ckpt_path=config.ckpt_load_path,
    ckpt_path=None,
  )

  trainer.test(
    model,
    dataloaders=dataloaders["test"],
  )


if __name__ == "__main__":
  main()
