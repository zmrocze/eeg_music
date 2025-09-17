"""Plotting utilities for EEG and music data visualization."""

from typing import Union, Dict, Any
import matplotlib.figure as mfig
from dataclasses import dataclass

from data import (
  TrialData,
  RawEeg,
  WavRAW,
  MelRaw,
  melspectrogram_figure,
  mkplot_melspectrogram,
)


@dataclass
class TrialPlots:
  """Container for trial visualization plots and metadata."""

  eeg_plot: mfig.Figure
  spectrogram_plot: mfig.Figure
  metadata: Dict[str, Any]


def plot_trial_data(trial_data: TrialData[RawEeg, Union[WavRAW, MelRaw]]) -> TrialPlots:
  """
  Create comprehensive plots for trial data including EEG and music spectrogram.

  Args:
      trial_data: TrialData containing RawEeg and either WavRAW or MelRaw music data

  Returns:
      TrialPlots containing EEG plot, spectrogram plot, and metadata
  """
  # Extract EEG data and create plot
  eeg_raw = trial_data.eeg_data.get_eeg().raw_eeg
  eeg_fig = eeg_raw.plot(show=False, title=f"EEG - {trial_data.trial_id}")
  music = trial_data.music_data.get_music()

  # Create spectrogram plot based on music data type
  match music:
    case WavRAW() as wav:
      spectrogram_fig = mkplot_melspectrogram(
        wav,
        title=f"Mel Spectrogram - {trial_data.music_filename.filename}",
        fmax=10240.0,
      )
    case MelRaw() as mel:
      # Use existing mel spectrogram
      spectrogram_fig = melspectrogram_figure(
        mel=mel, title=f"Mel Spectrogram - {trial_data.music_filename.filename}"
      )

  # Collect metadata
  metadata = {
    "dataset": trial_data.dataset,
    "subject": trial_data.subject,
    "session": trial_data.session,
    "run": trial_data.run,
    "trial_id": trial_data.trial_id,
    "music_filename": trial_data.music_filename.filename,
    "eeg_channels": eeg_raw.ch_names,
    "eeg_sample_rate": eeg_raw.info["sfreq"],
    "eeg_duration_seconds": eeg_raw.times[-1] if len(eeg_raw.times) > 0 else 0,
    "music_sample_rate": music.sample_rate,
    "music_duration_seconds": music.length_seconds(),
  }

  return TrialPlots(
    eeg_plot=eeg_fig, spectrogram_plot=spectrogram_fig, metadata=metadata
  )
