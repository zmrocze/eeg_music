
# data processing

on input:

1. training

 - EEG
  * 1000Hz
  * highpass 0Hz lowpass 500Hz
  * 37 channels


2. calibration

- EEG
  * 1000Hz
  * highpass 0Hz lowpass 500Hz
  * 37 channels


## preprocessing

1. downsample to 256Hz
2. cut to min length of music and eeg

## labram

trained on:

 - Remove irrelevant channels
 - Bandpass filter: 0.1–75 Hz
 - Notch filter: 50 Hz
 - Resample to 200 Hz
 - Set unit to µV

## cbramod

Channels: 19 standard EEG electrodes (10-20 system)
Sampling Rate: 200 Hz
Segment Length: 30 seconds (6000 samples)
Patch Size: 200 samples (1 second temporal window)
Data Format: Normalized EEG signals in units of 100 μV !!!!
Bandpass Filter: 0.3-75 Hz to remove low/high frequency noise
Notch Filter: 60 Hz to eliminate power line interference
Mask Ratio: 50% of patches randomly masked
Shape: [Batch_Size, 19_channels, 30_segments, 200_samples]

## eegpt

check finetuning parameters in paper: https://openreview.net/pdf?id=lvS2b8CjG5

I.e.: The training process is performed using AdamW optimizer, OneCycle learning rate strategy [ 43 ]
(starting learning rate 1.6e-5, maximum 4e-4, minimum 1.51e-7), 100 rounds of training, and batch
size of 64.

the downstream datasets are i.e.: 
physimi (25-50h)
seed (1000-4000 trials)

so comparable sizes, easier (classification) tasks.

4s trials: split to transformer into 64 250ms patches (50 samples?)

## labram

1s patch, variable max 256 patches
