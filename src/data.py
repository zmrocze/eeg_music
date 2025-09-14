"""Data types"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import (
  Dict,
  Union,
  Callable,
  TypeVar,
  Generic,
  List,
  TypedDict,
  Tuple,
  cast,
)
import numpy as np
from numpy.typing import NDArray
from mne.io import BaseRaw
import pandas as pd
from scipy.io import wavfile
import mne
from pandas import DataFrame, Index
import json
import shutil
import torch.utils.data as torchdata
from speechbrain.dataio.batch import PaddedBatch


class MusicData(ABC):
  """Abstract base class for music data."""

  @abstractmethod
  def get_music(self) -> "WavRAW":
    """Get the music as WavRAW data."""
    pass

  @abstractmethod
  def save(self, filepath: Path) -> None:
    """Save the music data to a file."""
    pass


class EegData(ABC):
  """Abstract base class for EEG data."""

  @abstractmethod
  def get_eeg(self) -> "RawEeg":
    """Get the EEG data as RawEeg."""
    pass

  @abstractmethod
  def save(self, filepath: Path) -> None:
    """Save the EEG data to a file."""
    pass


class MusicID(ABC):
  """Abstract base class for within-dataset music identifiers."""

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
  which_half: bool  # False: first half, True: second half of the music file

  def to_filename(self) -> str:
    """Convert training music ID to filename."""
    return f"{self.emotion_code_1}-{self.emotion_code_2}_{self.session}_{'second' if self.which_half else 'first'}.wav"


@dataclass
class WavRAW(MusicData):
  """Data class containing raw WAV data and its rate."""

  raw_data: NDArray[np.floating]  # Audio data as numpy array of float values
  sample_rate: int  # Sample rate in Hz

  def is_not_empty(self) -> bool:
    """Check if the WAV data is not empty."""
    return self.raw_data.size > 0

  def length_seconds(self) -> float:
    """Get the length of the WAV data in seconds."""
    return self.raw_data.shape[0] / self.sample_rate

  def get_music(self) -> "WavRAW":
    """Get the music as WavRAW data."""
    return self

  def save(self, filepath: Path) -> None:
    """Save the WAV data to a file."""
    wavfile.write(filepath, self.sample_rate, self.raw_data)


@dataclass
class OnDiskMusic(MusicData):
  """Music data backed by a file on disk."""

  filepath: Path

  def get_music(self) -> WavRAW:
    """Load and return the music as WavRAW data."""
    sample_rate, raw_data = wavfile.read(self.filepath)
    return WavRAW(raw_data=raw_data.astype(np.floating), sample_rate=sample_rate)

  def save(self, filepath: Path) -> None:
    """Save the music data by copying the file."""
    shutil.copy2(self.filepath, filepath)


@dataclass
class RawEeg(EegData):
  """EEG data stored in memory."""

  raw_eeg: BaseRaw

  def get_eeg(self) -> "RawEeg":
    """Get the EEG data."""
    return self

  def save(self, filepath: Path) -> None:
    """Save the EEG data to a file."""
    # mne.export.export_raw expects a Raw object, not a RawEeg wrapper
    mne.export.export_raw(filepath, self.raw_eeg, fmt="edf", overwrite=True)


@dataclass
class OnDiskEeg(EegData):
  """EEG data backed by a file on disk."""

  filepath: Path

  def get_eeg(self) -> "RawEeg":
    """Load and return the EEG data as RawEeg."""
    return RawEeg(raw_eeg=mne.io.read_raw(self.filepath, preload=False))

  def save(self, filepath: Path) -> None:
    """Save the EEG data by copying the file."""
    shutil.copy2(self.filepath, filepath)


# Type variables for generic Trial class
M = TypeVar("M", bound=MusicData)
E = TypeVar("E", bound=EegData)


@dataclass
class TrialData(Generic[E, M]):
  """Data class containing music and EEG data."""

  dataset: str
  subject: str
  session: str
  run: str
  trial_id: str

  eeg_data: E
  music_data: M


@dataclass(frozen=True)
class MusicFilename:
  """Reference to music data in the music collection."""

  filename: str

  @classmethod
  def from_musicid(cls, music_id: MusicID) -> "MusicFilename":
    """Create a MusicFilename from a MusicID."""
    return cls(filename=music_id.to_filename())


class TrialMetadataRecord(TypedDict):
  """Typed dict for trial metadata record in JSON."""

  dataset: str
  subject: str
  session: str
  run: str
  trial_id: str
  music_ref: str


@dataclass
class DatasetMetadata:
  """Metadata for a saved EEG music dataset: metadata.json."""

  trials: List[TrialMetadataRecord]
  stimuli: Dict[str, List[str]]  # Mapping from dataset name to list of music filenames
  num_trials: int

  def to_dict(self) -> dict:
    """Convert to dictionary for JSON serialization."""
    return {
      "trials": self.trials,
      "stimuli": self.stimuli,
      "num_trials": self.num_trials,
    }

  @classmethod
  def from_dict(cls, data: dict) -> "DatasetMetadata":
    """Create from dictionary loaded from JSON."""
    return cls(
      trials=data["trials"], stimuli=data["stimuli"], num_trials=data["num_trials"]
    )

  def save_json(self, filepath: Path) -> None:
    """Save metadata to JSON file."""
    with open(filepath, "w") as f:
      json.dump(self.to_dict(), f, indent=2)

  @classmethod
  def load_json(cls, filepath: Path) -> "DatasetMetadata":
    """Load metadata from JSON file."""
    with open(filepath, "r") as f:
      data = json.load(f)
    return cls.from_dict(data)


@dataclass
class TrialRow(Generic[E]):
  """Data class containing music ID, raw EEG data, and emotion code."""

  dataset: str
  subject: str
  session: str
  run: str
  trial_id: str
  eeg_data: E
  music_ref: MusicFilename


def make_eeg_path(
  base_dir: Path, dataset: str, subject: str, session: str, run: str, trial_id: str
) -> Path:
  """Construct the EEG file path following dataset/subject/session/run/trial_id/eeg.edf structure."""
  return base_dir / dataset / subject / session / run / trial_id / "eeg.edf"


def copy_from_dataloader_into_dir(loader, base_dir: Path):
  """
  Iterates over dataset loader trials and music collection,
  saving these into a specified directory.

  The directory can already contain a saved dataset.
  """

  base_dir.mkdir(parents=True, exist_ok=True)
  stimuli_dataset_dir = base_dir / "stimuli" / loader.dataset_name
  eeg_dir = base_dir / "eeg"
  stimuli_dataset_dir.mkdir(parents=True, exist_ok=True)
  eeg_dir.mkdir(exist_ok=True)

  metadata_path = base_dir / "metadata.json"
  existing_metadata = (
    DatasetMetadata.load_json(metadata_path)
    if metadata_path.exists()
    else DatasetMetadata(trials=[], stimuli={}, num_trials=0)
  )

  # Save music files
  for music_ref, music_data in loader.music_iterator():
    stimuli_file = stimuli_dataset_dir / music_ref.filename
    if not stimuli_file.exists():
      music_data.save(stimuli_file)
      if loader.dataset_name not in existing_metadata.stimuli:
        existing_metadata.stimuli[loader.dataset_name] = []
      if music_ref.filename not in existing_metadata.stimuli[loader.dataset_name]:
        existing_metadata.stimuli[loader.dataset_name].append(music_ref.filename)

  # Save trials
  for trial in loader.trial_iterator():
    trial_record: TrialMetadataRecord = {
      "dataset": trial.dataset,
      "subject": trial.subject,
      "session": trial.session,
      "run": trial.run,
      "trial_id": trial.trial_id,
      "music_ref": trial.music_ref.filename,
    }

    eeg_path = make_eeg_path(
      eeg_dir, trial.dataset, trial.subject, trial.session, trial.run, trial.trial_id
    )
    eeg_path.parent.mkdir(parents=True, exist_ok=True)

    # Save EEG data - all loaders now return EegData objects
    trial.eeg_data.save(eeg_path)

    existing_metadata.trials.append(trial_record)
    existing_metadata.num_trials += 1

  existing_metadata.save_json(metadata_path)


@dataclass(frozen=True)
class MusicRef:
  """Reference to music data in the full multi-dataset collection."""

  filename: MusicFilename
  dataset: str


class EEGMusicDataset(torchdata.Dataset):
  """
  Dataset containing EEG trials with metadata.

  Can be stored into a directory with structure:
  base_dir/
    metadata.json
    stimuli/
      dataset_name/
        music files...
    eeg/
      dataset_name/
        subject/
          session/
            run/
              trial_id/
                eeg.edf
  """

  def __init__(self):
    """
    Trial ids with eeg data (raw or on-disk) and music,
    pointed by reference into a music collection dict (which stores music raw or on-disk).
    That's because music files are often reused between trials.
    """
    self.df = pd.DataFrame(
      columns=Index(
        ["dataset", "subject", "session", "run", "trial_id", "music_ref", "eeg_data"]
      )
    )

    self.music_collection: Dict[MusicRef, MusicData] = {}

  @property
  def df(self) -> pd.DataFrame:
    """Get the dataframe."""
    return self._df

  @df.setter
  def df(self, value: pd.DataFrame) -> None:
    """Set the dataframe with proper indexing."""
    cols = ["dataset", "subject", "session", "run", "trial_id"]
    indexed = value.reindex(columns=cols + ["music_ref", "eeg_data"])
    indexed = indexed.set_index(
      cols,
      drop=False,
      verify_integrity=True,
    )
    self._df = indexed

  def __len__(self) -> int:
    return len(self.df)

  def __getitem__(self, idx: int) -> TrialData[EegData, MusicData]:
    row = self.df.iloc[idx]
    music_ref = MusicRef(filename=row.music_ref, dataset=row.dataset)
    music_data = self.music_collection[music_ref]
    return TrialData(
      dataset=row.dataset,
      subject=row.subject,
      session=row.session,
      run=row.run,
      trial_id=row.trial_id,
      eeg_data=row.eeg_data,
      music_data=music_data,
    )

  def merge(self, other: "EEGMusicDataset") -> "EEGMusicDataset":
    """Merge this dataset with another dataset."""
    merged_dataset = EEGMusicDataset()
    merged_dataset.df = pd.concat([self.df, other.df], ignore_index=True)
    # this is enough because music refs are unique outside of dataset as well
    merged_dataset.music_collection = {
      **self.music_collection,
      **other.music_collection,
    }
    return merged_dataset

  def map_df(self, func: Callable[[pd.DataFrame], pd.DataFrame]) -> "EEGMusicDataset":
    """Map underlying dataframe."""
    mapped_dataset = EEGMusicDataset()
    df = func(self.df.copy())
    mapped_dataset.df = df
    return mapped_dataset

  def subject_wise_split(
    self, p_train: float, p_val: float, seed: int = 42
  ) -> Tuple["EEGMusicDataset", "EEGMusicDataset", "EEGMusicDataset"]:
    """Split subjects into train/val/test using two proportions.

    p_train: fraction of subjects for train
    p_val: fraction of subjects for val (after train); must satisfy p_train>0, p_val>=0, p_train+p_val<1
    Remainder subjects form test. Deterministic via seed.
    """
    if not (0 < p_train < 1):
      raise ValueError("p_train in (0,1)")
    if not (0 <= p_val < 1):
      raise ValueError("p_val in [0,1)")
    if p_train + p_val >= 1:
      raise ValueError("p_train + p_val < 1 required")
    np.random.seed(seed)
    subj = np.array(self.df["subject"].unique())
    np.random.shuffle(subj)
    n = len(subj)
    n_tr = int(n * p_train)
    n_va = int(n * p_val)

    def mk(s: np.ndarray) -> "EEGMusicDataset":
      ds = EEGMusicDataset()
      df = self.df[self.df["subject"].isin(s.tolist())]
      ds.df = cast(DataFrame, df.reset_index(drop=True))
      ds.music_collection = self.music_collection
      return ds

    return mk(subj[:n_tr]), mk(subj[n_tr : n_tr + n_va]), mk(subj[n_tr + n_va :])

  def save(self, base_dir: Path) -> None:
    """Save dataset to directory with metadata and trial data."""
    base_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = base_dir / "metadata.json"

    if metadata_path.exists():
      print(f"Overwriting existing metadata at {metadata_path}")

    # Create stimuli directories and save music
    stimuli_dir = base_dir / "stimuli"
    for music_ref, music_data in self.music_collection.items():
      music_dir = stimuli_dir / music_ref.dataset
      music_dir.mkdir(parents=True, exist_ok=True)
      music_data.save(music_dir / music_ref.filename.filename)

    # Save EEG data
    eeg_dir = base_dir / "eeg"
    for _, row in self.df.iterrows():
      eeg_path = make_eeg_path(
        eeg_dir, row.dataset, row.subject, row.session, row.run, row.trial_id
      )
      eeg_path.parent.mkdir(parents=True, exist_ok=True)
      row.eeg_data.save(eeg_path)

    # Save metadata
    stimuli_by_dataset = {}
    for music_ref in self.music_collection.keys():
      if music_ref.dataset not in stimuli_by_dataset:
        stimuli_by_dataset[music_ref.dataset] = []
      stimuli_by_dataset[music_ref.dataset].append(music_ref.filename.filename)

    metadata = DatasetMetadata(
      trials=[
        {
          "dataset": row.dataset,
          "subject": row.subject,
          "session": row.session,
          "run": row.run,
          "trial_id": row.trial_id,
          "music_ref": row.music_ref.filename,
        }
        for _, row in self.df.iterrows()
      ],
      stimuli=stimuli_by_dataset,
      num_trials=len(self.df),
    )
    metadata.save_json(metadata_path)

  @classmethod
  def load_ondisk(cls, base_dir: Path) -> "EEGMusicDataset":
    """Load dataset from directory, using music and EEG data on-disk representations."""
    dataset = cls()

    # Read metadata
    metadata = DatasetMetadata.load_json(base_dir / "metadata.json")

    # Create music collection from metadata stimuli
    stimuli_dir = base_dir / "stimuli"
    for dataset_name, music_filenames in metadata.stimuli.items():
      for music_filename in music_filenames:
        music_file_path = stimuli_dir / dataset_name / music_filename
        music_ref = MusicRef(
          filename=MusicFilename(filename=music_filename), dataset=dataset_name
        )
        dataset.music_collection[music_ref] = OnDiskMusic(filepath=music_file_path)

    # Create dataframe from trial metadata
    eeg_dir = base_dir / "eeg"
    rows = []
    for trial_record in metadata.trials:
      eeg_path = make_eeg_path(
        eeg_dir,
        trial_record["dataset"],
        trial_record["subject"],
        trial_record["session"],
        trial_record["run"],
        trial_record["trial_id"],
      )
      rows.append(
        {
          "dataset": trial_record["dataset"],
          "subject": trial_record["subject"],
          "session": trial_record["session"],
          "run": trial_record["run"],
          "trial_id": trial_record["trial_id"],
          "music_ref": MusicFilename(filename=trial_record["music_ref"]),
          "eeg_data": OnDiskEeg(filepath=eeg_path),
        }
      )

    dataset.df = pd.DataFrame(rows)
    return dataset

  def load_to_mem(self):
    """Load all eeg and music data into memory."""
    # Convert all MusicData to WavRAW in the music collection
    for music_ref, music_data in self.music_collection.items():
      self.music_collection[music_ref] = music_data.get_music()

    # Convert all EegData to RawEeg in the dataframe
    for idx, row in self.df.iterrows():
      self.df.at[idx, "eeg_data"] = row.eeg_data.get_eeg()

  def remove_short_trials(self, min_trial_length_seconds: float) -> "EEGMusicDataset":
    """Return a new dataset with trials shorter than the threshold removed.

    A trial is kept iff both:
    - EEG duration in seconds >= min_trial_length_seconds
    - Music duration in seconds >= min_trial_length_seconds
    """
    to_keep: List[int] = []
    for i in range(len(self)):
      trial = self[i]
      raw = trial.eeg_data.get_eeg().raw_eeg
      sfreq = float(raw.info["sfreq"]) if "sfreq" in raw.info else raw.info["sfreq"]
      eeg_duration_sec = raw.n_times / sfreq
      if (
        eeg_duration_sec >= min_trial_length_seconds
        and trial.music_data.get_music().length_seconds() >= min_trial_length_seconds
      ):
        to_keep.append(i)
    filtered = EEGMusicDataset()
    filtered.df = self.df.iloc[to_keep].reset_index(drop=True)
    filtered.music_collection = self.music_collection
    return filtered


def example_collate_fn(trials: List[TrialData[EegData, MusicData]]):
  # todo: preload before collate_fn? matters with pin_memory and all that?
  eegs = [t.eeg_data.get_eeg() for t in trials]
  music = [t.music_data.get_music() for t in trials]
  return PaddedBatch(eegs), PaddedBatch(music)


def prepare_trial(trial: TrialData[EegData, MusicData]):
  """Set common length between music and eeg, resample eeg to 256Hz."""

  eeg: BaseRaw = trial.eeg_data.get_eeg().raw_eeg
  m_len = trial.music_data.get_music().length_seconds()
  e_len = eeg.n_times / eeg.info["sfreq"]
  min_len = min(m_len, e_len)

  music = trial.music_data.get_music()
  music = WavRAW(
    music.raw_data[: int(min_len * music.sample_rate)], sample_rate=music.sample_rate
  )

  eeg: BaseRaw = cast(BaseRaw, eeg.copy().resample(256))
  eeg = eeg.crop(
    tmax=(min(min_len, eeg.times[-1]))
  )  # when l=e_len then eeg_times[-1] is that 1s/sample_rate early to l which errors

  return TrialData(
    dataset=trial.dataset,
    subject=trial.subject,
    session=trial.session,
    run=trial.run,
    trial_id=trial.trial_id,
    eeg_data=RawEeg(raw_eeg=eeg),
    music_data=music,
  )


class MappedDataset(EEGMusicDataset):
  """Dataset with a mapping function applied to each trial on access."""

  def __init__(
    self,
    base_dataset: EEGMusicDataset,
    map_fn: Callable[[TrialData[EegData, MusicData]], TrialData[E, M]],
  ):
    super().__init__()
    self.df = base_dataset.df
    self.music_collection = base_dataset.music_collection
    self.map_fn = map_fn  # type: ignore[assignment]

  def __getitem__(self, idx: int) -> TrialData[EegData, MusicData]:
    trial = super().__getitem__(idx)
    return cast(TrialData[EegData, MusicData], self.map_fn(trial))
