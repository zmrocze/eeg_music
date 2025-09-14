
## Dataset

training + calibration is ~28h of eeg recordings
"overlapped" over 3h of generated synthetic music

## considerations

for comparison:
GTZAN music dataset is ~8h of music (1000 tracks of 30s)
and Musimple  DiT can generate 80 bands × 800 frames mel-spectrograms
(divides into 8x8 patches which are then treated as single tokens)
trains with single-GPU (e.g., NVIDIA RTX 4090) with batch size 48, and 100 000 training steps (they provide pretrained for 2 days on 4090).


a day of a100 costs ~80$ !!!
l4, t4 ~20$

plan: defer training for as long as colab credits last
Q: how long do these last?

## training details

### optimizer

AdamW optimizer?

First learning rate finder: 
 - https://pytorch-lightning.readthedocs.io/en/1.5.10/advanced/lr_finder.html

Then maybe CosineAnnealingWarmRestarts or OneCycleLR (if known num of epochs)
cosineannealing with T_0 \in [5, 200] (10k to 100k samples dataset?)
and T_mul=2 for cycle length doubling every cycle
eta_min = optimal_lr / 100   # Conservative minimum

### data loader

i.e.:
prefetch_factor=1 or max 2, (with many workers it should be enough to have it be 1) also benchmark because maybe we dont do any augmentations (??)
num_workers>1
persistent_workers=True,
pin_memory=True # if enough ram

use non_blocking=True when moving data for async batch loading

can use collate_fn to i.e. set to fixed tensor size (but maybe better in preprocessing, we can do that, snippets should be mostly same size. or bucketized to differ 21 vs 19 etc)

Bucketized Batch Sampling ??

### wb

https://docs.wandb.ai/guides/runs/alert/

## metrics

https://www.perplexity.ai/search/brief-story-of-reagan-presiden-eiFslgL7R.maMgyHf_SPEA

https://pypi.org/project/Audio-Similarity/

https://lightning.ai/docs/torchmetrics/stable/audio/perceptual_evaluation_speech_quality.html
