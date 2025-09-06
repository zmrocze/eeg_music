"""Tests for EEGMusicDataset save/load functionality."""

import unittest
import tempfile
import numpy as np
from pathlib import Path

from data import EEGMusicDataset
from bcmi import BCMICalibrationLoader, BCMITrainingLoader


class TestDatasetPersistence(unittest.TestCase):
  """Test EEGMusicDataset save/load functionality with BCMI data."""

  @classmethod
  def setUpClass(cls):
    """Set up class-level test data."""
    cls.bcmi_path = Path("/home/zmrocze/studia/uwr/magisterka/datasets/bcmi")

  def create_test_dataset(self) -> EEGMusicDataset:
    """Load calibration and training datasets from BCMI data."""
    combined_dataset = EEGMusicDataset()

    # Load training data
    training_path = self.bcmi_path / "bcmi-training"
    if training_path.exists():
      training_loader = BCMITrainingLoader(str(training_path))
      training_loader.load_all_subjects(max_subjects=1, max_runs_per_session=1)
      for trial in training_loader.trial_iterator():
        combined_dataset.add_trial(trial)

    # Load calibration data
    calibration_path = self.bcmi_path / "bcmi-calibration"
    if calibration_path.exists():
      calibration_loader = BCMICalibrationLoader(str(calibration_path))
      calibration_loader.load_all_subjects(max_subjects=1, max_runs_per_session=1)
      for trial in calibration_loader.trial_iterator():
        combined_dataset.add_trial(trial)

    return combined_dataset

  def assert_datasets_equal(self, ds1: EEGMusicDataset, ds2: EEGMusicDataset) -> None:
    """Assert that two datasets are equivalent, providing detailed failure messages."""
    # Check basic structure
    self.assertEqual(
      len(ds1.df),
      len(ds2.df),
      f"Dataset lengths differ: {len(ds1.df)} vs {len(ds2.df)}",
    )

    # Check metadata columns
    for col in ["dataset", "subject", "session", "run"]:
      self.assertTrue(
        ds1.df[col].equals(ds2.df[col]), f"Column '{col}' differs between datasets"
      )

    # Check trial data (compare loaded data)
    for idx in range(len(ds1.df)):
      data1 = ds1.df.iloc[idx]["trial_data"]
      data2 = ds2.df.iloc[idx]["trial_data"]

      # Compare audio
      np.testing.assert_allclose(
        data1.get_music_raw().raw_data,
        data2.get_music_raw().raw_data,
        err_msg=f"Audio data differs for trial {idx}",
      )
      self.assertEqual(
        data1.get_music_raw().sample_rate,
        data2.get_music_raw().sample_rate,
        f"Audio sample rate differs for trial {idx}: {data1.get_music_raw().sample_rate} vs {data2.get_music_raw().sample_rate}",
      )

      # Compare EEG (allow for EDF format conversion precision loss)
      # Either 0.01% relative difference OR 0.01 absolute difference is acceptable
      np.testing.assert_allclose(
        data1.get_eeg_raw().get_data(),
        data2.get_eeg_raw().get_data(),
        rtol=0.0001,
        atol=0.01,
        err_msg=f"EEG data differs for trial {idx}",
      )

  def test_dataset_save_load_with_ondisk(self):
    """Test dataset save/load with load_ondisk intermediate step."""
    # Create original dataset
    original_dataset = self.create_test_dataset()

    with tempfile.TemporaryDirectory() as temp_dir:
      temp_path = Path(temp_dir) / "test_dataset"

      # Save dataset
      original_dataset.save(temp_path)

      # Load back as OnDisk dataset, then convert to memory
      loaded_ondisk = EEGMusicDataset.load_ondisk(temp_path)
      loaded_inmem = loaded_ondisk.load_to_mem()

      # Verify datasets are equivalent
      self.assert_datasets_equal(original_dataset, loaded_inmem)

      print("✓ Save/load with ondisk intermediate step successful")

  def test_dataset_save_load_direct(self):
    """Test dataset save/load without load_ondisk intermediate step."""
    # Create original dataset
    original_dataset = self.create_test_dataset()

    with tempfile.TemporaryDirectory() as temp_dir:
      temp_path = Path(temp_dir) / "test_dataset"

      # Save dataset
      original_dataset.save(temp_path)

      # Load back and immediately convert to memory (skip ondisk step)
      loaded = EEGMusicDataset.load_ondisk(temp_path)

      # Verify datasets are equivalent
      self.assert_datasets_equal(original_dataset, loaded)

      print("✓ Save/load direct (no intermediate step) successful")


if __name__ == "__main__":
  unittest.main(verbosity=2)
