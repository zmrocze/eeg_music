"""Data types"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Union
import numpy as np
from numpy.typing import NDArray
from mne.io import BaseRaw
import pandas as pd
from scipy.io import wavfile
import mne


class MusicID(ABC):
  """Abstract base class for music identifiers."""

  @abstractmethod
  def to_filename(self) -> str:
    """Convert the music ID to a filename string."""
    pass


wav_filenames_ordered_calibration = [
  "hvha1.wav",
  "hvha10.wav",
  "hvha11.wav",
  "hvha12.wav",
  "hvha2.wav",
  "hvha3.wav",
  "hvha4.wav",
  "hvha5.wav",
  "hvha6.wav",
  "hvha7.wav",
  "hvha8.wav",
  "hvha9.wav",
  "hvla1.wav",
  "hvla10.wav",
  "hvla11.wav",
  "hvla12.wav",
  "hvla2.wav",
  "hvla3.wav",
  "hvla4.wav",
  "hvla5.wav",
  "hvla6.wav",
  "hvla7.wav",
  "hvla8.wav",
  "hvla9.wav",
  "hvna1.wav",
  "hvna10.wav",
  "hvna11.wav",
  "hvna12.wav",
  "hvna2.wav",
  "hvna3.wav",
  "hvna4.wav",
  "hvna5.wav",
  "hvna6.wav",
  "hvna7.wav",
  "hvna8.wav",
  "hvna9.wav",
  "lvha1.wav",
  "lvha10.wav",
  "lvha11.wav",
  "lvha12.wav",
  "lvha2.wav",
  "lvha3.wav",
  "lvha4.wav",
  "lvha5.wav",
  "lvha6.wav",
  "lvha7.wav",
  "lvha8.wav",
  "lvha9.wav",
  "lvla1.wav",
  "lvla10.wav",
  "lvla11.wav",
  "lvla12.wav",
  "lvla2.wav",
  "lvla3.wav",
  "lvla4.wav",
  "lvla5.wav",
  "lvla6.wav",
  "lvla7.wav",
  "lvla8.wav",
  "lvla9.wav",
  "lvna1.wav",
  "lvna10.wav",
  "lvna11.wav",
  "lvna12.wav",
  "lvna2.wav",
  "lvna3.wav",
  "lvna4.wav",
  "lvna5.wav",
  "lvna6.wav",
  "lvna7.wav",
  "lvna8.wav",
  "lvna9.wav",
  "nvha1.wav",
  "nvha10.wav",
  "nvha11.wav",
  "nvha12.wav",
  "nvha2.wav",
  "nvha3.wav",
  "nvha4.wav",
  "nvha5.wav",
  "nvha6.wav",
  "nvha7.wav",
  "nvha8.wav",
  "nvha9.wav",
  "nvla1.wav",
  "nvla10.wav",
  "nvla11.wav",
  "nvla12.wav",
  "nvla2.wav",
  "nvla3.wav",
  "nvla4.wav",
  "nvla5.wav",
  "nvla6.wav",
  "nvla7.wav",
  "nvla8.wav",
  "nvla9.wav",
  "nvna1.wav",
  "nvna10.wav",
  "nvna11.wav",
  "nvna12.wav",
  "nvna2.wav",
  "nvna3.wav",
  "nvna4.wav",
  "nvna5.wav",
  "nvna6.wav",
  "nvna7.wav",
  "nvna8.wav",
  "nvna9.wav",
]

# !!!!! These files are 19s long, not 21s !!!!!


@dataclass
class CalibrationMusicId(MusicID):
  """Music ID for calibration data."""

  number: int

  def to_filename(self) -> str:
    """Convert calibration music ID to filename."""
    return wav_filenames_ordered_calibration[self.number]


@dataclass
class TrainingMusicId(MusicID):
  """Music ID for training data."""

  emotion_code_1: int
  emotion_code_2: int
  session: Union[int, str]

  def to_filename(self) -> str:
    """Convert training music ID to filename."""
    return f"{self.emotion_code_1}-{self.emotion_code_2}_{self.session}.wav"


@dataclass
class WavRAW:
  """Data class containing raw WAV data and its rate."""

  raw_data: NDArray[np.floating]  # Audio data as numpy array of float values
  sample_rate: int  # Sample rate in Hz

  def is_not_empty(self) -> bool:
    """Check if the WAV data is not empty."""
    return self.raw_data.size > 0


class TrialData(ABC):
  """Abstract base class for trial data."""

  @abstractmethod
  def get_music_raw(self) -> WavRAW:
    """Get the music raw data."""
    pass

  @abstractmethod
  def get_eeg_raw(self) -> BaseRaw:
    """Get the EEG raw data."""
    pass


@dataclass
class RawTrial(TrialData):
  """Trial data stored in memory."""

  music_raw: WavRAW
  raw_eeg: BaseRaw

  def get_music_raw(self) -> WavRAW:
    """Get the music raw data."""
    return self.music_raw

  def get_eeg_raw(self) -> BaseRaw:
    """Get the EEG raw data."""
    return self.raw_eeg


@dataclass
class OnDiskTrial(TrialData):
  """Trial data referenced by file paths."""

  music_file_path: Path
  eeg_file_path: Path

  def get_music_raw(self) -> WavRAW:
    """Load and return the music raw data from file."""
    sample_rate, data = wavfile.read(self.music_file_path)
    return WavRAW(raw_data=data, sample_rate=sample_rate)

  def get_eeg_raw(self) -> BaseRaw:
    """Load and return the EEG raw data from file."""
    raw = mne.io.read_raw_bdf(self.eeg_file_path, preload=True)
    return raw


@dataclass
class Trial:
  """Data class containing music ID, raw EEG data, and emotion code."""

  dataset: str
  subject: str
  session: str
  run: str
  data: TrialData


class Dataset:
  """Dataset containing EEG trials with metadata."""

  def __init__(self, records=None):
    if records is None:
      # Use Index to fix type issue with columns parameter
      from pandas import Index

      self.df = pd.DataFrame(
        columns=Index(["subject", "run", "session", "trial_id", "trial"])
      )
    else:
      data = []
      for record in records:
        data.append(
          {
            "subject": record.subject,
            "run": record.run,
            "session": record.session,
            "trial_id": record.trial_id,
            "trial": record,
          }
        )
      self.df = pd.DataFrame(data)

  def add_trial(
    self, subject: str, run: int, session: int, trial_id: int, trial: Trial
  ):
    """Add a trial to the dataset."""
    new_row = pd.DataFrame(
      {
        "subject": [subject],
        "run": [run],
        "session": [session],
        "trial_id": [trial_id],
        "trial": [trial],
      }
    )
    self.df = pd.concat([self.df, new_row], ignore_index=True)

  def merge(self, other: "Dataset") -> "Dataset":
    """Merge this dataset with another dataset."""
    merged_dataset = Dataset()
    merged_dataset.df = pd.concat([self.df, other.df], ignore_index=True)
    return merged_dataset

  def map(self, func) -> "Dataset":
    """Map underlying dataframe."""
    mapped_dataset = Dataset()
    df = func(self.df.copy())
    mapped_dataset.df = df
    return mapped_dataset
