"""Data types"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Union, Callable, cast
import numpy as np
from numpy.typing import NDArray
from mne.io import BaseRaw
import pandas as pd
from scipy.io import wavfile
import mne
from pandas import Index
import json
import shutil


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

  @abstractmethod
  def save(self, base_dir: Path) -> None:
    """Save trial data to directory."""
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

  def save(self, base_dir: Path) -> None:
    """Save trial to directory using BIDS format for EEG."""
    eeg_dir = base_dir / "eeg"
    audio_dir = base_dir / "audio"
    eeg_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(exist_ok=True)

    mne.export.export_raw(eeg_dir / "eeg.edf", self.raw_eeg, fmt="edf", overwrite=False)
    wavfile.write(
      audio_dir / "audio.wav", self.music_raw.sample_rate, self.music_raw.raw_data
    )


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
    raw = mne.io.read_raw(self.eeg_file_path, preload=True)
    return raw

  def load(self) -> RawTrial:
    """Load trial data from file paths and return a RawTrial."""
    return RawTrial(music_raw=self.get_music_raw(), raw_eeg=self.get_eeg_raw())

  def save(self, base_dir: Path) -> None:
    """Save trial data by copying files to target directory."""
    eeg_dir = base_dir / "eeg"
    audio_dir = base_dir / "audio"
    eeg_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(exist_ok=True)

    shutil.copy2(self.eeg_file_path, eeg_dir / "eeg.edf")
    shutil.copy2(self.music_file_path, audio_dir / "audio.wav")


@dataclass
class Trial:
  """Data class containing music ID, raw EEG data, and emotion code."""

  dataset: str
  subject: str
  session: str
  run: str
  trial_id: str
  data: TrialData


class EEGMusicDataset:
  """Dataset containing EEG trials with metadata."""

  def __init__(self, records=None):
    if records is None:
      self.df = pd.DataFrame(
        columns=Index(
          ["dataset", "subject", "session", "run", "trial_id", "trial_data"]
        )
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

  def add_trial(self, trial: Trial):
    """Add a trial to the dataset."""
    new_row = pd.DataFrame(
      {
        "dataset": [trial.dataset],
        "subject": [trial.subject],
        "session": [trial.session],
        "run": [trial.run],
        "trial_id": [trial.trial_id],
        "trial_data": [trial.data],
      }
    )
    self.df = pd.concat([self.df, new_row], ignore_index=True)

  def merge(self, other: "EEGMusicDataset") -> "EEGMusicDataset":
    """Merge this dataset with another dataset."""
    merged_dataset = EEGMusicDataset()
    merged_dataset.df = pd.concat([self.df, other.df], ignore_index=True)
    return merged_dataset

  def map(self, func: Callable[[pd.DataFrame], pd.DataFrame]) -> "EEGMusicDataset":
    """Map underlying dataframe."""
    mapped_dataset = EEGMusicDataset()
    df = func(self.df.copy())
    mapped_dataset.df = df
    return mapped_dataset

  def save(self, base_dir: Path) -> None:
    """Save dataset to directory with metadata and trial data."""
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    # Save metadata
    metadata = []
    for idx, row in self.df.iterrows():
      trial_dir = f"{row['dataset']}_{row['subject']}_{row['session']}_{row['run']}_{row['trial_id']}"
      metadata.append(
        {
          "dataset": row["dataset"],
          "subject": row["subject"],
          "session": row["session"],
          "run": row["run"],
          "trial_id": row["trial_id"],
          "trial_dir": trial_dir,
        }
      )

      # Save trial data
      trial_data: TrialData = cast(TrialData, row["trial_data"])
      trial_data.save(base_dir / trial_dir)

    with open(base_dir / "metadata.json", "w") as f:
      json.dump(metadata, f, indent=2)

  @classmethod
  def load_ondisk(cls, base_dir: Path) -> "EEGMusicDataset":
    """Load dataset from directory."""
    base_dir = Path(base_dir)

    with open(base_dir / "metadata.json") as f:
      metadata = json.load(f)

    dataset = cls()
    for item in metadata:
      trial_dir = base_dir / item["trial_dir"]
      trial_data = OnDiskTrial(
        music_file_path=trial_dir / "audio" / "audio.wav",
        eeg_file_path=trial_dir / "eeg" / "eeg.edf",
      )

      trial = Trial(
        dataset=item["dataset"],
        subject=item["subject"],
        session=item["session"],
        run=item["run"],
        trial_id=item["trial_id"],
        data=trial_data,
      )
      dataset.add_trial(trial)

    return dataset

  def load_to_mem(self):
    """Load all trial data into memory, converting OnDiskTrial to RawTrial."""

    def load_trial_data(df):
      return df.assign(
        trial_data=df["trial_data"].map(
          lambda data: data.load() if isinstance(data, OnDiskTrial) else data
        )
      )

    return self.map(load_trial_data)
