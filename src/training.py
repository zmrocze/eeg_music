from diffusers import DiffusionPipeline
import torch

# from prefigure.prefigure import push_wandb_config
from torch import optim
import pytorch_lightning as pl
import torchtune.training
import wandb
import torchtune
from datasets import load_from_disk
from dataclasses import dataclass

# import torch_audiomentations as taug
import random

# import diffusers
from pytorch_lightning.callbacks import RichProgressBar


def todo():
  raise NotImplementedError("This needs to be todoed")


@dataclass
class TrainingConfig:
  model_name = "unlocked-250k"
  data_path = "./rock_dataset_resampled"
  eeg_sample_size = todo()
  music_sample_rate = todo()
  data_loader_num_workers = 4
  batch_size = 8
  # val_batch_size = 4
  lr = 4e-5
  lr_warmup_steps = 5  # epochs
  T_mul = 2
  num_cycles = 0.5  # cosine annealing cycles
  num_epochs = 100
  # save_image_epochs = 10
  save_model_per_epochs = 30
  val_every = 5
  # save_demo = True
  ckpt_load_path = None  # 'best', 'last', <path]>
  wandb_log_model = "all"
  check_val_every_n_epoch = 5
  # mixed_precision = "fp16"  # `no` for float32, `fp16` for automatic mixed precision
  output_dir = f"ddim-lora-{model_name}"  # the model name locally and on the HF Hub
  randint = random.randint(0, 1000)
  name = f"ddim-lora-{model_name}-{randint}"  # the name of the wandb run
  project_name = "neural-music-decoding"
  save_path = f"{name}-ckpt"
  # overwrite_output_dir = True  # overwrite the old model when re-running the notebook
  # seed = 42


config = TrainingConfig()


def optional(x, bool):
  if bool:
    return [x]
  else:
    return []


def count_n_params(model):
  """Counts the number of trainable parameters in a model."""
  return sum(p.numel() for p in model.parameters() if p.requires_grad)


class EegptSpectrogram(pl.LightningModule):
  def __init__(self, config):  # todo: config
    super().__init__()
    self.config = config
    self.rng = torch.quasirandom.SobolEngine(1, scramble=True, seed=42)

  def configure_optimizers(self):
    optimizer = optim.AdamW(self.model.parameters(), lr=config.lr)
    #  torchtune.modules.get_cosine_schedule_with_warmup(optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int, num_cycles: float = 0.5, last_epoch: int = - 1) → LambdaLR[source]
    lr_schedule = torchtune.training.get_cosine_schedule_with_warmup(
      optimizer,
      num_warmup_steps=config.lr_warmup_steps,
      num_training_steps=config.num_epochs,
      num_cycles=config.num_cycles,
      last_epoch=-1,
    )
    return {
      "optimizer": optimizer,
      "lr_scheduler": {
        "scheduler": lr_schedule,
        "interval": "epoch",  # or "epoch"
        "frequency": 1,
      },
    }

  # @property
  # def model(self):
  #   return self.pipe.unet

  def log_some_samples(self, val_x, val_y):
    "Log images to wandb and save them to disk"

    # val_x: (batch, audio_len) (unused here), val_y: (batch, freq, time) spectrograms
    specs = val_y.detach().cpu()
    images = []
    for i, spec in enumerate(specs):
      s = spec.float()
      s = (s - s.min()) / (s.max() - s.min() + 1e-8)
      images.append(wandb.Image(s.numpy(), caption=f"spectrogram_{i}"))
    self.logger.experiment.log({"val/spectrograms": images}, step=self.global_step)

  def training_step(self, batch, batch_idx):
    # noise, noise_pred = self.random_timestep_forward(batch)
    # loss = F.mse_loss(noise_pred, noise)
    loss = self.forward_dance_diffusion(batch)
    self.log_dict(
      {"train/loss": loss.detach()},
      prog_bar=True,
      on_step=True,
      on_epoch=True,
      batch_size=self.config.batch_size,
    )

    return loss

  def validation_step(self, batch, batch_idx):
    # noise, noise_pred = self.random_timestep_forward(batch)
    # loss = F.mse_loss(noise_pred, noise)
    loss = self.forward_dance_diffusion(batch)
    self.log_dict({"val/loss": loss.detach()}, prog_bar=True)

    return loss

  def metrics(self):
    pass


class DemoCallback(pl.Callback):
  def __init__(self, config):
    super().__init__()
    self.demo_every = config.demo_every
    self.n_samples = config.n_samples

  def on_train_start(self, trainer, pl_module):
    print("Logging some initial samples...")
    pl_module.log_some_samples(trainer.global_step, self.n_samples)

  def on_train_epoch_end(self, trainer, pl_module):
    if (trainer.current_epoch + 1) % self.demo_every == 0:
      pl_module.log_some_samples(trainer.global_step, self.n_samples)


def main(config=config):
  device = "cuda" if torch.cuda.is_available() else "cpu"
  data = load_from_disk(config.data_path)
  data_loaders = make_dataloaders(data, config)
  pipe = DiffusionPipeline.from_pretrained(f"harmonai/{config.model_name}")
  pipe.to(device)
  wandb_logger = pl.loggers.WandbLogger(
    project=config.project_name,
    name=config.name,
    log_model=config.wandb_log_model,
    config=config,
  )
  save_on_exc = pl.callbacks.OnExceptionCheckpoint(
    f"{config.save_path}/exc_save",
  )
  ckpt_callback = pl.callbacks.ModelCheckpoint(
    every_n_epochs=1,
    save_on_train_epoch_end=False,  # together with every_n_epochs and check_val_every_n_epoch, this will save the model on validation, that is on check_val_every_n_epoch
    auto_insert_metric_name=True,
    mode="min",
    monitor="val/loss",
    save_top_k=2,
    dirpath=config.save_path,
    save_last=True,
  )
  demo_callback = DemoCallback(config)

  apply_lora(pipe.unet, config)
  model = DiffusionUncond(pipe, config)

  wandb_logger.watch(model, log="all")

  trainer = pl.Trainer(
    # precision=16,
    accumulate_grad_batches=config.gradient_accumulation_steps,
    callbacks=[ckpt_callback, demo_callback, RichProgressBar(), save_on_exc],
    logger=wandb_logger,
    check_val_every_n_epoch=config.check_val_every_n_epoch,
    # log_every_n_steps=1,
    max_epochs=config.num_epochs,
  )

  print(f"Model.unet trainable params: {count_n_params(model.model)}")

  trainer.fit(
    model,
    train_dataloaders=data_loaders["train"],
    val_dataloaders=data_loaders["validation"],
    ckpt_path=config.ckpt_load_path,
  )


if __name__ == "__main__":
  main()
