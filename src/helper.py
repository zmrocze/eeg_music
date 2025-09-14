import numpy as np
import librosa
import librosa.display as lbd
import matplotlib.pyplot as plt
from data import WavRAW


def onset_secs_to_samples(onset_secs, sfreq):
  return round(onset_secs * sfreq)


def wavraw_to_melspectrogram(
  wav: WavRAW,
  n_mels: int = 128,
  n_fft: int | None = None,
  hop_length: int | None = None,
  fmin: float = 0.0,
  fmax: float | None = None,
  center: bool = True,
  power: float = 2.0,
  to_db: bool = True,
) -> np.ndarray:
  """Return (log) mel-spectrogram for a WavRAW.

  Defaults: 128 mels, n_fft=2048, hop=512, power spectrogram, dB-scaled.
  """
  y = wav.raw_data if wav.raw_data.ndim == 1 else np.mean(wav.raw_data, axis=1)
  y = y.astype(np.float32)
  m = np.max(np.abs(y))
  y = y / (m + 1e-12) if m > 1.0 else y
  S = librosa.feature.melspectrogram(
    y=y,
    sr=wav.sample_rate,
    n_fft=2048 if n_fft is None else n_fft,
    hop_length=512 if hop_length is None else hop_length,
    n_mels=n_mels,
    fmin=fmin,
    fmax=fmax,
    center=center,
    power=power,
    norm="slaney",
    htk=False,
  )
  return librosa.power_to_db(S, ref=np.max) if to_db else S


def melspectrogram_figure(
  spec: np.ndarray,
  sample_rate: int,
  fmin: float = 0.0,
  fmax: float | None = None,
  to_db: bool = True,
  cmap: str = "magma",
  title: str = "Mel-spectrogram",
):
  """Build and return a matplotlib Figure with the mel-spectrogram plot."""
  S = spec
  fig, ax = plt.subplots(figsize=(8, 3))
  img = lbd.specshow(
    S,
    x_axis="time",
    y_axis="mel",
    sr=sample_rate,
    fmin=fmin,
    fmax=fmax,
    cmap=cmap,
    ax=ax,
  )
  ax.set(title=title + (" (dB)" if to_db else ""))
  cbar = fig.colorbar(img, ax=ax)
  cbar.set_label("dB" if to_db else "power")
  fig.tight_layout()
  return fig


def plot_melspectrogram(
  wav: WavRAW,
  **kwargs,
):
  """Plot the mel-spectrogram and show it. Returns the created Figure."""
  fig = melspectrogram_figure(
    wavraw_to_melspectrogram(wav, **kwargs),
    sample_rate=wav.sample_rate,
    **kwargs,
  )
  plt.show()
  return fig
