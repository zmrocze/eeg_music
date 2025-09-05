"""
Test for data types and Trial save/load functionality.
"""

import unittest
import tempfile
from pathlib import Path
import numpy as np
import mne
from scipy.io import wavfile


from data import Trial, OnDiskTrial, RawTrial, WavRAW


class TestTrialSaveLoad(unittest.TestCase):
  """Test Trial save and load functionality with real BCMI data."""

  @classmethod
  def setUpClass(cls):
    """Set up class-level test data."""
    # Path to BCMI training dataset
    cls.bcmi_path = Path(
      "/home/zmrocze/studia/uwr/magisterka/datasets/bcmi/bcmi-training"
    )

    # Check if dataset exists
    if not cls.bcmi_path.exists():
      raise unittest.SkipTest("BCMI training dataset not found")

    # Find first available subject and session
    cls.eeg_file: Path | None = None
    cls.audio_file: Path | None = None

    for subject_dir in cls.bcmi_path.glob("sub-*"):
      for session_dir in subject_dir.glob("ses-*"):
        eeg_dir = session_dir / "eeg"
        if eeg_dir.exists():
          # Find first EEG file
          eeg_files = list(eeg_dir.glob("*_eeg.edf"))
          if eeg_files:
            cls.eeg_file = eeg_files[0]

            # Find corresponding audio file in stimuli
            stimuli_dir = cls.bcmi_path / "stimuli"
            audio_files = list(stimuli_dir.glob("*.wav"))
            if audio_files:
              cls.audio_file = audio_files[0]
              break
      if cls.eeg_file and cls.audio_file:
        break

    if not cls.eeg_file or not cls.audio_file:
      raise unittest.SkipTest(
        "Could not find suitable EEG and audio files in BCMI dataset"
      )

    print(f"Using EEG file: {cls.eeg_file}")
    print(f"Using audio file: {cls.audio_file}")

  def test_save_and_load_roundtrip(self):
    """Test that EEG data remains unchanged after save/load roundtrip."""

    # Ensure files are available (should be set by setUpClass)
    assert self.eeg_file is not None, "EEG file not found"
    assert self.audio_file is not None, "Audio file not found"

    # Create trial with real BCMI data
    trial_data = OnDiskTrial(
      music_file_path=self.audio_file, eeg_file_path=self.eeg_file
    )

    # Load original EEG data
    original_raw = trial_data.get_eeg_raw()
    original_data = original_raw.get_data()
    original_info = original_raw.info

    # Ensure original_data is numpy array
    assert isinstance(original_data, np.ndarray), (
      f"Expected ndarray, got {type(original_data)}"
    )

    print(f"Original EEG shape: {original_data.shape}")
    print(f"Original sampling rate: {original_info['sfreq']} Hz")
    print(f"Original channels: {len(original_info['ch_names'])}")

    # Save trial to temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
      base_path = Path(temp_dir)

      # Load the trial data into memory first and save it
      raw_trial_data = trial_data.load()  # Use the new load method
      raw_trial_data.save(base_path)  # Use the new save method

      # Check files were created
      eeg_file = base_path / "eeg" / "eeg.edf"
      audio_file = base_path / "audio" / "audio.wav"

      self.assertTrue(eeg_file.exists(), "EEG file was not created")
      self.assertTrue(audio_file.exists(), "Audio file was not created")

      # Load the saved EEG data back
      loaded_raw = mne.io.read_raw_edf(eeg_file, preload=True)
      loaded_data = loaded_raw.get_data()
      loaded_info = loaded_raw.info

      # Ensure loaded_data is numpy array
      assert isinstance(loaded_data, np.ndarray), (
        f"Expected ndarray, got {type(loaded_data)}"
      )

      print(f"Loaded EEG shape: {loaded_data.shape}")
      print(f"Loaded sampling rate: {loaded_info['sfreq']} Hz")
      print(f"Loaded channels: {len(loaded_info['ch_names'])}")

      # Compare data shapes
      self.assertEqual(
        original_data.shape, loaded_data.shape, "EEG data shape changed after save/load"
      )

      # Compare sampling rates
      self.assertEqual(
        original_info["sfreq"],
        loaded_info["sfreq"],
        "Sampling rate changed after save/load",
      )

      # Compare number of channels
      self.assertEqual(
        len(original_info["ch_names"]),
        len(loaded_info["ch_names"]),
        "Number of channels changed after save/load",
      )

      # Compare data values (allowing for small numerical differences due to format conversion)
      # EDF format conversion can introduce small precision losses
      np.testing.assert_allclose(
        original_data,
        loaded_data,
        rtol=1e-6,
        atol=1e-7,
        err_msg="EEG data values changed after save/load",
      )

      print("✓ EEG data integrity verified - roundtrip successful!")

      # Also test audio data
      original_audio = trial_data.get_music_raw()
      loaded_audio_rate, loaded_audio_data = wavfile.read(audio_file)

      self.assertEqual(
        original_audio.sample_rate,
        loaded_audio_rate,
        "Audio sample rate changed after save/load",
      )

      # Convert to same type for comparison
      original_audio_data = original_audio.raw_data.astype(loaded_audio_data.dtype)
      np.testing.assert_array_equal(
        original_audio_data,
        loaded_audio_data,
        err_msg="Audio data changed after save/load",
      )

      print("✓ Audio data integrity verified - roundtrip successful!")


class TestOnDiskTrial(unittest.TestCase):
  """Test OnDiskTrial functionality with real BCMI dataset files."""

  @classmethod
  def setUpClass(cls):
    """Find suitable test files from BCMI datasets."""
    cls.bcmi_training_path = Path(
      "/home/zmrocze/studia/uwr/magisterka/datasets/bcmi/bcmi-training"
    )
    cls.bcmi_calibration_path = Path(
      "/home/zmrocze/studia/uwr/magisterka/datasets/bcmi/bcmi-calibration"
    )

    # Find files from training dataset
    cls.training_eeg_file = None
    cls.training_audio_file = None

    if cls.bcmi_training_path.exists():
      # Find first available EEG file
      for subject_dir in cls.bcmi_training_path.glob("sub-*"):
        for session_dir in subject_dir.glob("ses-*"):
          eeg_dir = session_dir / "eeg"
          if eeg_dir.exists():
            eeg_files = list(eeg_dir.glob("*_eeg.edf"))
            if eeg_files:
              cls.training_eeg_file = eeg_files[0]
              break
        if cls.training_eeg_file:
          break

      # Find audio file from stimuli
      stimuli_dir = cls.bcmi_training_path / "stimuli"
      if stimuli_dir.exists():
        audio_files = list(stimuli_dir.glob("*.wav"))
        if audio_files:
          cls.training_audio_file = audio_files[0]

    # Find files from calibration dataset
    cls.calibration_eeg_file = None
    cls.calibration_audio_file = None

    if cls.bcmi_calibration_path.exists():
      # Find first available EEG file (calibration has no ses-* subdirs, just sub-*/eeg/)
      for subject_dir in cls.bcmi_calibration_path.glob("sub-*"):
        eeg_dir = subject_dir / "eeg"  # No session directory in calibration
        if eeg_dir.exists():
          eeg_files = list(eeg_dir.glob("*_eeg.edf"))
          if eeg_files:
            cls.calibration_eeg_file = eeg_files[0]
            break

      # Find audio file from stimuli
      stimuli_dir = cls.bcmi_calibration_path / "stimuli"
      if stimuli_dir.exists():
        audio_files = list(stimuli_dir.glob("*.wav"))
        if audio_files:
          cls.calibration_audio_file = audio_files[0]

  def test_ondisktrial_basic_functionality(self):
    """Test basic OnDiskTrial functionality with training data."""
    if not self.training_eeg_file or not self.training_audio_file:
      self.skipTest("BCMI training dataset files not found")

    print(f"Testing with EEG file: {self.training_eeg_file}")
    print(f"Testing with audio file: {self.training_audio_file}")

    # Create OnDiskTrial
    trial_data = OnDiskTrial(
      music_file_path=self.training_audio_file, eeg_file_path=self.training_eeg_file
    )

    # Test EEG loading
    eeg_raw = trial_data.get_eeg_raw()
    self.assertIsInstance(eeg_raw, mne.io.BaseRaw, "Should return MNE BaseRaw object")
    self.assertGreater(len(eeg_raw.ch_names), 0, "Should have channels")
    self.assertGreater(len(eeg_raw.times), 0, "Should have time samples")
    print(
      f"✓ EEG loaded: {len(eeg_raw.ch_names)} channels, {len(eeg_raw.times)} samples"
    )
    print(f"✓ EEG sampling rate: {eeg_raw.info['sfreq']} Hz")
    print(f"✓ EEG duration: {eeg_raw.times[-1]:.1f} seconds")

    # Test music loading
    music_raw = trial_data.get_music_raw()
    self.assertIsInstance(music_raw, WavRAW, "Should return WavRAW object")
    self.assertTrue(music_raw.is_not_empty(), "Music data should not be empty")
    self.assertGreater(music_raw.sample_rate, 0, "Should have valid sample rate")
    print(
      f"✓ Audio loaded: {music_raw.raw_data.shape} samples at {music_raw.sample_rate} Hz"
    )
    print(
      f"✓ Audio duration: {len(music_raw.raw_data) / music_raw.sample_rate:.1f} seconds"
    )

  def test_ondisktrial_with_calibration_data(self):
    """Test OnDiskTrial with calibration dataset."""
    if not self.calibration_eeg_file or not self.calibration_audio_file:
      self.skipTest("BCMI calibration dataset files not found")

    print(f"Testing with calibration EEG file: {self.calibration_eeg_file}")
    print(f"Testing with calibration audio file: {self.calibration_audio_file}")

    # Create OnDiskTrial
    trial_data = OnDiskTrial(
      music_file_path=self.calibration_audio_file,
      eeg_file_path=self.calibration_eeg_file,
    )

    # Test loading
    eeg_raw = trial_data.get_eeg_raw()
    music_raw = trial_data.get_music_raw()

    self.assertIsInstance(eeg_raw, mne.io.BaseRaw)
    self.assertIsInstance(music_raw, WavRAW)
    self.assertTrue(music_raw.is_not_empty())

    print("✓ Calibration data loaded successfully")
    print(f"✓ EEG: {len(eeg_raw.ch_names)} channels, {eeg_raw.times[-1]:.1f}s")
    print(f"✓ Audio: {len(music_raw.raw_data) / music_raw.sample_rate:.1f}s")

  def test_ondisktrial_in_trial_object(self):
    """Test OnDiskTrial as part of Trial object."""
    if not self.training_eeg_file or not self.training_audio_file:
      self.skipTest("BCMI training dataset files not found")

    # Create Trial with OnDiskTrial data
    trial_data = OnDiskTrial(
      music_file_path=self.training_audio_file, eeg_file_path=self.training_eeg_file
    )

    trial = Trial(
      dataset="bcmi-training", subject="test", session="1", run="1", data=trial_data
    )

    # Test that we can access data through Trial
    eeg_raw = trial.data.get_eeg_raw()
    music_raw = trial.data.get_music_raw()

    self.assertIsInstance(eeg_raw, mne.io.BaseRaw)
    self.assertIsInstance(music_raw, WavRAW)

    # Test save functionality
    with tempfile.TemporaryDirectory() as temp_dir:
      base_path = Path(temp_dir)

      # Load trial data into memory and save it
      raw_trial_data = trial_data.load()  # Convert OnDiskTrial to RawTrial
      raw_trial_data.save(base_path)  # Save using RawTrial.save method

      # Check files were created
      eeg_file = base_path / "eeg" / "eeg.edf"
      audio_file = base_path / "audio" / "audio.wav"

      self.assertTrue(eeg_file.exists(), "EEG file should be saved")
      self.assertTrue(audio_file.exists(), "Audio file should be saved")

      print("✓ OnDiskTrial data successfully saved through RawTrial.save method")

  def test_ondisktrial_lazy_loading(self):
    """Test that OnDiskTrial loads data lazily (on demand)."""
    if not self.training_eeg_file or not self.training_audio_file:
      self.skipTest("BCMI training dataset files not found")

    # Create OnDiskTrial - should not load data yet
    trial_data = OnDiskTrial(
      music_file_path=self.training_audio_file, eeg_file_path=self.training_eeg_file
    )

    # Verify paths are stored correctly
    self.assertEqual(trial_data.music_file_path, self.training_audio_file)
    self.assertEqual(trial_data.eeg_file_path, self.training_eeg_file)

    # Now load data - should work on demand
    eeg_raw = trial_data.get_eeg_raw()
    music_raw = trial_data.get_music_raw()

    # Load again - should work multiple times
    eeg_raw2 = trial_data.get_eeg_raw()
    music_raw2 = trial_data.get_music_raw()

    # Data should be equivalent (though not necessarily identical objects)
    np.testing.assert_array_equal(eeg_raw.get_data(), eeg_raw2.get_data())
    np.testing.assert_array_equal(music_raw.raw_data, music_raw2.raw_data)
    self.assertEqual(music_raw.sample_rate, music_raw2.sample_rate)

    print("✓ OnDiskTrial lazy loading works correctly")

  def test_ondisktrial_file_not_found_errors(self):
    """Test OnDiskTrial error handling for missing files."""
    # Test with non-existent files
    trial_data = OnDiskTrial(
      music_file_path=Path("/nonexistent/audio.wav"),
      eeg_file_path=Path("/nonexistent/eeg.edf"),
    )

    # Should raise appropriate errors when trying to load
    with self.assertRaises(FileNotFoundError):
      trial_data.get_music_raw()

    # For EEG files, MNE might raise different exceptions
    with self.assertRaises((FileNotFoundError, OSError, ValueError, Exception)):
      trial_data.get_eeg_raw()

    print("✓ OnDiskTrial properly handles missing files")


class TestRawTrialSave(unittest.TestCase):
  """Test RawTrial save functionality."""

  @classmethod
  def setUpClass(cls):
    """Set up test data for RawTrial tests."""
    cls.bcmi_path = Path(
      "/home/zmrocze/studia/uwr/magisterka/datasets/bcmi/bcmi-training"
    )

    # Find test files
    cls.eeg_file = None
    cls.audio_file = None

    if cls.bcmi_path.exists():
      for subject_dir in cls.bcmi_path.glob("sub-*"):
        for session_dir in subject_dir.glob("ses-*"):
          eeg_dir = session_dir / "eeg"
          if eeg_dir.exists():
            eeg_files = list(eeg_dir.glob("*_eeg.edf"))
            if eeg_files:
              cls.eeg_file = eeg_files[0]
              break
        if cls.eeg_file:
          break

      stimuli_dir = cls.bcmi_path / "stimuli"
      if stimuli_dir.exists():
        audio_files = list(stimuli_dir.glob("*.wav"))
        if audio_files:
          cls.audio_file = audio_files[0]

  def test_rawtrial_save_basic(self):
    """Test basic RawTrial save functionality."""
    if not self.eeg_file or not self.audio_file:
      self.skipTest("BCMI dataset files not found")

    # Load data into memory first
    on_disk_trial = OnDiskTrial(
      music_file_path=self.audio_file, eeg_file_path=self.eeg_file
    )
    raw_trial = on_disk_trial.load()

    # Verify we have a RawTrial
    self.assertIsInstance(raw_trial, RawTrial)
    self.assertIsInstance(raw_trial.music_raw, WavRAW)
    self.assertIsInstance(raw_trial.raw_eeg, mne.io.BaseRaw)

    # Test save
    with tempfile.TemporaryDirectory() as temp_dir:
      base_path = Path(temp_dir)
      raw_trial.save(base_path)

      # Check files were created
      eeg_file = base_path / "eeg" / "eeg.edf"
      audio_file = base_path / "audio" / "audio.wav"

      self.assertTrue(eeg_file.exists(), "EEG file should be saved")
      self.assertTrue(audio_file.exists(), "Audio file should be saved")

      print("✓ RawTrial.save() creates files correctly")

  def test_rawtrial_save_data_integrity(self):
    """Test that RawTrial.save preserves data integrity."""
    if not self.eeg_file or not self.audio_file:
      self.skipTest("BCMI dataset files not found")

    # Create RawTrial
    on_disk_trial = OnDiskTrial(
      music_file_path=self.audio_file, eeg_file_path=self.eeg_file
    )
    raw_trial = on_disk_trial.load()

    # Store original data
    original_eeg_data = raw_trial.raw_eeg.get_data()
    original_music_data = raw_trial.music_raw.raw_data
    original_sample_rate = raw_trial.music_raw.sample_rate

    # Save and reload
    with tempfile.TemporaryDirectory() as temp_dir:
      base_path = Path(temp_dir)
      raw_trial.save(base_path)

      # Load saved files
      eeg_file = base_path / "eeg" / "eeg.edf"
      audio_file = base_path / "audio" / "audio.wav"

      loaded_eeg = mne.io.read_raw_edf(eeg_file, preload=True)
      loaded_sample_rate, loaded_music_data = wavfile.read(audio_file)

      # Compare data
      np.testing.assert_allclose(
        original_eeg_data, loaded_eeg.get_data(), rtol=1e-6, atol=1e-7
      )
      self.assertEqual(original_sample_rate, loaded_sample_rate)
      np.testing.assert_array_equal(
        original_music_data.astype(loaded_music_data.dtype), loaded_music_data
      )

      print("✓ RawTrial.save() preserves data integrity")


class TestOnDiskTrialLoad(unittest.TestCase):
  """Test OnDiskTrial load functionality."""

  @classmethod
  def setUpClass(cls):
    """Set up test data for OnDiskTrial load tests."""
    cls.bcmi_path = Path(
      "/home/zmrocze/studia/uwr/magisterka/datasets/bcmi/bcmi-training"
    )

    # Find test files
    cls.eeg_file = None
    cls.audio_file = None

    if cls.bcmi_path.exists():
      for subject_dir in cls.bcmi_path.glob("sub-*"):
        for session_dir in subject_dir.glob("ses-*"):
          eeg_dir = session_dir / "eeg"
          if eeg_dir.exists():
            eeg_files = list(eeg_dir.glob("*_eeg.edf"))
            if eeg_files:
              cls.eeg_file = eeg_files[0]
              break
        if cls.eeg_file:
          break

      stimuli_dir = cls.bcmi_path / "stimuli"
      if stimuli_dir.exists():
        audio_files = list(stimuli_dir.glob("*.wav"))
        if audio_files:
          cls.audio_file = audio_files[0]

  def test_ondisktrial_load_returns_rawtrial(self):
    """Test that OnDiskTrial.load() returns a RawTrial instance."""
    if not self.eeg_file or not self.audio_file:
      self.skipTest("BCMI dataset files not found")

    on_disk_trial = OnDiskTrial(
      music_file_path=self.audio_file, eeg_file_path=self.eeg_file
    )

    raw_trial = on_disk_trial.load()

    # Verify return type
    self.assertIsInstance(raw_trial, RawTrial)
    self.assertIsInstance(raw_trial.music_raw, WavRAW)
    self.assertIsInstance(raw_trial.raw_eeg, mne.io.BaseRaw)

    print("✓ OnDiskTrial.load() returns RawTrial instance")

  def test_ondisktrial_load_data_equivalence(self):
    """Test that OnDiskTrial.load() produces equivalent data to direct access."""
    if not self.eeg_file or not self.audio_file:
      self.skipTest("BCMI dataset files not found")

    on_disk_trial = OnDiskTrial(
      music_file_path=self.audio_file, eeg_file_path=self.eeg_file
    )

    # Get data through direct methods
    direct_eeg = on_disk_trial.get_eeg_raw()
    direct_music = on_disk_trial.get_music_raw()

    # Get data through load method
    raw_trial = on_disk_trial.load()
    loaded_eeg = raw_trial.get_eeg_raw()
    loaded_music = raw_trial.get_music_raw()

    # Compare EEG data
    np.testing.assert_array_equal(direct_eeg.get_data(), loaded_eeg.get_data())
    self.assertEqual(direct_eeg.info["sfreq"], loaded_eeg.info["sfreq"])

    # Compare music data
    np.testing.assert_array_equal(direct_music.raw_data, loaded_music.raw_data)
    self.assertEqual(direct_music.sample_rate, loaded_music.sample_rate)

    print("✓ OnDiskTrial.load() produces equivalent data to direct methods")

  def test_ondisktrial_load_multiple_calls(self):
    """Test that OnDiskTrial.load() can be called multiple times."""
    if not self.eeg_file or not self.audio_file:
      self.skipTest("BCMI dataset files not found")

    on_disk_trial = OnDiskTrial(
      music_file_path=self.audio_file, eeg_file_path=self.eeg_file
    )

    # Load multiple times
    raw_trial1 = on_disk_trial.load()
    raw_trial2 = on_disk_trial.load()

    # Should be separate instances
    self.assertIsNot(raw_trial1, raw_trial2)

    # But with equivalent data
    np.testing.assert_array_equal(
      raw_trial1.raw_eeg.get_data(), raw_trial2.raw_eeg.get_data()
    )
    np.testing.assert_array_equal(
      raw_trial1.music_raw.raw_data, raw_trial2.music_raw.raw_data
    )
    self.assertEqual(raw_trial1.music_raw.sample_rate, raw_trial2.music_raw.sample_rate)

    print("✓ OnDiskTrial.load() can be called multiple times")


class TestLoadSaveRoundtrip(unittest.TestCase):
  """Test complete load/save roundtrip: OnDiskTrial -> RawTrial -> save -> load."""

  @classmethod
  def setUpClass(cls):
    """Set up test data."""
    cls.bcmi_path = Path(
      "/home/zmrocze/studia/uwr/magisterka/datasets/bcmi/bcmi-training"
    )

    # Find test files
    cls.eeg_file = None
    cls.audio_file = None

    if cls.bcmi_path.exists():
      for subject_dir in cls.bcmi_path.glob("sub-*"):
        for session_dir in subject_dir.glob("ses-*"):
          eeg_dir = session_dir / "eeg"
          if eeg_dir.exists():
            eeg_files = list(eeg_dir.glob("*_eeg.edf"))
            if eeg_files:
              cls.eeg_file = eeg_files[0]
              break
        if cls.eeg_file:
          break

      stimuli_dir = cls.bcmi_path / "stimuli"
      if stimuli_dir.exists():
        audio_files = list(stimuli_dir.glob("*.wav"))
        if audio_files:
          cls.audio_file = audio_files[0]

  def test_complete_roundtrip(self):
    """Test: OnDiskTrial -> load() -> RawTrial -> save() -> OnDiskTrial -> load()."""
    if not self.eeg_file or not self.audio_file:
      self.skipTest("BCMI dataset files not found")

    # Start with OnDiskTrial
    original_on_disk = OnDiskTrial(
      music_file_path=self.audio_file, eeg_file_path=self.eeg_file
    )

    # Load into memory
    raw_trial = original_on_disk.load()

    # Get original data
    original_eeg_data = raw_trial.raw_eeg.get_data()
    original_music_data = raw_trial.music_raw.raw_data
    original_sample_rate = raw_trial.music_raw.sample_rate

    # Save to disk
    with tempfile.TemporaryDirectory() as temp_dir:
      base_path = Path(temp_dir)
      raw_trial.save(base_path)

      # Create new OnDiskTrial from saved files
      saved_on_disk = OnDiskTrial(
        music_file_path=base_path / "audio" / "audio.wav",
        eeg_file_path=base_path / "eeg" / "eeg.edf",
      )

      # Load again
      reloaded_raw_trial = saved_on_disk.load()

      # Compare final data with original
      reloaded_eeg_data = reloaded_raw_trial.raw_eeg.get_data()
      reloaded_music_data = reloaded_raw_trial.music_raw.raw_data
      reloaded_sample_rate = reloaded_raw_trial.music_raw.sample_rate

      # EEG comparison (allow for format conversion precision loss)
      np.testing.assert_allclose(
        original_eeg_data, reloaded_eeg_data, rtol=1e-6, atol=1e-7
      )

      # Audio comparison
      self.assertEqual(original_sample_rate, reloaded_sample_rate)
      np.testing.assert_array_equal(
        original_music_data.astype(reloaded_music_data.dtype), reloaded_music_data
      )

      print(
        "✓ Complete OnDiskTrial -> RawTrial -> save -> OnDiskTrial roundtrip successful"
      )


if __name__ == "__main__":
  unittest.main(verbosity=2)
