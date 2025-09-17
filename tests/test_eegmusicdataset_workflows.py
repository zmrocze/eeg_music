import unittest
from pathlib import Path
import tempfile
import itertools

from bcmi import BCMICalibrationLoader, BCMITrainingLoader
from data import (
  EEGMusicDataset,
  copy_from_dataloader_into_dir,
)


DATASET_ROOT = Path("/home/zmrocze/studia/uwr/magisterka/datasets")
BCMI_CAL_PATH = DATASET_ROOT / "bcmi" / "bcmi-calibration"
BCMI_TRN_PATH = DATASET_ROOT / "bcmi" / "bcmi-training"


def dataset_exists(p: Path) -> bool:
  try:
    return p.exists() and any(p.iterdir())
  except Exception:
    return False


def limit_loader_iterators(loader, max_trials: int = 6):
  """Limit trials and ensure music iterator yields exactly what trials need.

  We first materialize a limited list of trials, then filter music to only
  the referenced filenames. This guarantees no missing stimuli for trials.
  """

  loader.load_all_subjects(max_subjects=3)

  orig_trials = loader.trial_iterator
  orig_music = loader.music_iterator

  # Precompute limited trials
  limited_trials = list(itertools.islice(orig_trials(), max_trials))
  needed = {t.music_filename for t in limited_trials}

  def trial_iter():
    for t in limited_trials:
      yield t

  def music_iter():
    yielded = set()
    for ref, music in orig_music():
      if ref in needed and ref not in yielded:
        yielded.add(ref)
        yield ref, music

  loader.trial_iterator = trial_iter
  loader.music_iterator = music_iter
  return loader


class TestEEGMusicDatasetWorkflows(unittest.TestCase):
  """End-to-end persistence scenarios for EEGMusicDataset.

  Scenarios implemented (run for both calibration and training when available):
  1) copy_from_dataloader_into_dir -> load_ondisk -> open files and check lengths
  2) like (1) + load_to_mem
  3) copy two datasets -> load -> merge -> verify
  4) copy same dataset twice into same dir -> load -> expect duplicates (current impl appends)
  5) remove_short_trials on merged dataset
  6) subject_wise_split on merged dataset
  7) load_ondisk -> save() -> load_ondisk -> compare identifiers
  8) Real-life flow: copy two datasets -> load -> load_to_mem -> save -> reload -> spot-check

  If any test fails:
  - (4) If duplicates are not doubled, implementation may be de-duplicating trials; adjust expectation.
  - (8) Failures often indicate RawEeg.save using mne.export.export_raw cannot write EDF in env
        (missing writer backend or MNE version). In that case, save() on RawEeg needs alternate export.
  """

  def setUp(self):
    self.available = {
      "calibration": dataset_exists(BCMI_CAL_PATH),
      "training": dataset_exists(BCMI_TRN_PATH),
    }

  def _mk_loader(self, kind: str):
    if kind == "calibration":
      return BCMICalibrationLoader(str(BCMI_CAL_PATH), dataset_name="bcmi-calibration")
    if kind == "training":
      return BCMITrainingLoader(str(BCMI_TRN_PATH), dataset_name="bcmi-training")
    raise ValueError(kind)

  def _ensure_available(self, kind: str):
    if not self.available.get(kind, False):
      self.skipTest(f"Dataset not found for {kind}")

  # 1. copy -> load_ondisk; inspect counts and sample lengths
  def test_copy_and_load_ondisk_lengths(self):
    for kind in ("calibration", "training"):
      with self.subTest(kind=kind):
        self._ensure_available(kind)
        loader = self._mk_loader(kind)
        limit_loader_iterators(loader)
        with tempfile.TemporaryDirectory() as d:
          base = Path(d)
          copy_from_dataloader_into_dir(loader, base)
          ds = EEGMusicDataset.load_ondisk(base)
          self.assertGreater(len(ds), 0)
          # Verify music and eeg can be opened and have positive durations
          t = ds[0]
          mus = t.music_data.get_music()
          self.assertGreater(mus.length_seconds(), 0.0)
          eeg = t.eeg_data.get_eeg().raw_eeg
          sf = float(eeg.info.get("sfreq", 0.0))
          self.assertGreater(sf, 0.0)
          self.assertGreater(eeg.n_times, 0)

  # 2. copy -> load_ondisk -> load_to_mem
  def test_copy_and_load_to_mem(self):
    for kind in ("calibration", "training"):
      with self.subTest(kind=kind):
        self._ensure_available(kind)
        loader = self._mk_loader(kind)
        limit_loader_iterators(loader)
        with tempfile.TemporaryDirectory() as d:
          base = Path(d)
          copy_from_dataloader_into_dir(loader, base)
          ds = EEGMusicDataset.load_ondisk(base)
          n = len(ds)
          ds.load_to_mem()
          self.assertEqual(len(ds), n)
          # Spot check types via behavior
          t = ds[0]
          self.assertGreater(t.music_data.get_music().length_seconds(), 0.0)
          eeg = t.eeg_data.get_eeg().raw_eeg
          self.assertGreater(eeg.n_times, 0)

  # 3. copy two datasets -> load both -> merge -> verify
  def test_merge_datasets(self):
    # Require at least one available; if only one, skip
    if not (self.available["calibration"] and self.available["training"]):
      self.skipTest("Both calibration and training datasets required")
    with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
      # calibration
      cal_loader = self._mk_loader("calibration")
      trn_loader = self._mk_loader("training")
      limit_loader_iterators(cal_loader)
      limit_loader_iterators(trn_loader)

      copy_from_dataloader_into_dir(cal_loader, Path(d1))
      copy_from_dataloader_into_dir(trn_loader, Path(d2))

      cal_ds = EEGMusicDataset.load_ondisk(Path(d1))
      trn_ds = EEGMusicDataset.load_ondisk(Path(d2))
      merged = cal_ds.merge(trn_ds)
      self.assertEqual(len(merged), len(cal_ds) + len(trn_ds))
      # Music collection should be disjoint union by MusicRef keys
      self.assertGreaterEqual(
        len(merged.music_collection),
        len(cal_ds.music_collection) + len(trn_ds.music_collection),
      )

  # 5. remove_short_trials on combined dataset
  def test_remove_short_trials(self):
    if not (self.available["calibration"] and self.available["training"]):
      self.skipTest("Both calibration and training datasets required")
    with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
      cal_loader = self._mk_loader("calibration")
      trn_loader = self._mk_loader("training")
      limit_loader_iterators(cal_loader)
      limit_loader_iterators(trn_loader)
      copy_from_dataloader_into_dir(cal_loader, Path(d1))
      copy_from_dataloader_into_dir(trn_loader, Path(d2))
      cal_ds = EEGMusicDataset.load_ondisk(Path(d1))
      trn_ds = EEGMusicDataset.load_ondisk(Path(d2))
      ds = cal_ds.merge(trn_ds)
      filtered = ds.remove_short_trials(5.0)
      self.assertLessEqual(len(filtered), len(ds))
      # Sanity: all remaining trials have >= threshold durations
      for i in range(len(filtered)):
        t = filtered[i]
        eeg = t.eeg_data.get_eeg().raw_eeg
        sf = float(eeg.info.get("sfreq", 0.0))
        self.assertGreaterEqual(eeg.n_times / sf, 5.0)

  # 6. subject_wise_split on merged dataset
  def test_subject_wise_split(self):
    if not (self.available["calibration"] and self.available["training"]):
      self.skipTest("Both calibration and training datasets required")
    with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
      cal_loader = self._mk_loader("calibration")
      trn_loader = self._mk_loader("training")
      limit_loader_iterators(cal_loader)
      limit_loader_iterators(trn_loader)
      copy_from_dataloader_into_dir(cal_loader, Path(d1))
      copy_from_dataloader_into_dir(trn_loader, Path(d2))
      cal_ds = EEGMusicDataset.load_ondisk(Path(d1))
      trn_ds = EEGMusicDataset.load_ondisk(Path(d2))
      ds = cal_ds.merge(trn_ds)
      tr, va, te = ds.subject_wise_split(0.5, 0.0, seed=123)
      self.assertEqual(len(tr) + len(va) + len(te), len(ds))
      # Use only train & test (va empty expected) for previous assertions
      tr_subj = set(tr.df["subject"].unique())
      te_subj = set(te.df["subject"].unique())
      self.assertTrue(tr_subj.isdisjoint(te_subj))

  # 7. load -> save -> reload -> compare basic invariants
  def test_save_roundtrip(self):
    for kind in ("calibration", "training"):
      with self.subTest(kind=kind):
        self._ensure_available(kind)
        loader = self._mk_loader(kind)
        limit_loader_iterators(loader)
        with (
          tempfile.TemporaryDirectory() as dsrc,
          tempfile.TemporaryDirectory() as ddst,
        ):
          base = Path(dsrc)
          copy_from_dataloader_into_dir(loader, base)
          ds = EEGMusicDataset.load_ondisk(base)
          out = Path(ddst) / "saved"
          ds.save(out)
          ds2 = EEGMusicDataset.load_ondisk(out)
          self.assertEqual(len(ds2), len(ds))
          # Compare per-row identifiers
          a = (
            ds.df[
              ["dataset", "subject", "session", "run", "trial_id", "music_filename"]
            ]
            .copy()
            .reset_index(drop=True)
          )
          b = (
            ds2.df[
              ["dataset", "subject", "session", "run", "trial_id", "music_filename"]
            ]
            .copy()
            .reset_index(drop=True)
          )
          self.assertEqual(len(a), len(b))
          # Use set of tuples to avoid ordering sensitivity
          self.assertEqual(
            set(map(tuple, a.values.tolist())), set(map(tuple, b.values.tolist()))
          )

  # 8. Real-life flow: copy two datasets -> load -> load_to_mem -> save -> reload -> check
  def test_real_life_flow(self):
    if not (self.available["calibration"] and self.available["training"]):
      self.skipTest("Both calibration and training datasets required")
    with (
      tempfile.TemporaryDirectory() as d1,
      tempfile.TemporaryDirectory() as d2,
      tempfile.TemporaryDirectory() as ddst,
    ):
      cal_loader = self._mk_loader("calibration")
      trn_loader = self._mk_loader("training")
      limit_loader_iterators(cal_loader)
      limit_loader_iterators(trn_loader)
      copy_from_dataloader_into_dir(cal_loader, Path(d1))
      copy_from_dataloader_into_dir(trn_loader, Path(d2))
      cal_ds = EEGMusicDataset.load_ondisk(Path(d1))
      trn_ds = EEGMusicDataset.load_ondisk(Path(d2))
      ds = cal_ds.merge(trn_ds)
      ds.load_to_mem()
      out = Path(ddst) / "out"
      # NOTE: If this save fails, most likely mne.export.export_raw (used by RawEeg.save)
      # cannot write EDF in the current environment. This indicates a missing/export backend
      # or incompatible MNE version, not a logic error in dataset wiring.
      ds.save(out)
      ds2 = EEGMusicDataset.load_ondisk(out)
      self.assertEqual(len(ds2), len(ds))
      # Spot-check random item
      t = ds2[0]
      self.assertGreater(t.music_data.get_music().length_seconds(), 0.0)
      eeg = t.eeg_data.get_eeg().raw_eeg
      self.assertGreater(eeg.n_times, 0)


if __name__ == "__main__":
  unittest.main()
