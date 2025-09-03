"""Data types"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Union, Tuple
from mne.io import BaseRaw
import numpy as np


class MusicID(ABC):
    """Abstract base class for music identifiers."""
    
    @abstractmethod
    def to_filename(self) -> str:
        """Convert the music ID to a filename string."""
        pass

wav_filenames_ordered_calibration = [
    'hvha1.wav', 'hvha10.wav', 'hvha11.wav', 'hvha12.wav', 'hvha2.wav', 'hvha3.wav', 'hvha4.wav', 
    'hvha5.wav', 'hvha6.wav', 'hvha7.wav', 'hvha8.wav', 'hvha9.wav', 'hvla1.wav', 'hvla10.wav', 
    'hvla11.wav', 'hvla12.wav', 'hvla2.wav', 'hvla3.wav', 'hvla4.wav', 'hvla5.wav', 'hvla6.wav',
    'hvla7.wav', 'hvla8.wav', 'hvla9.wav', 'hvna1.wav', 'hvna10.wav', 'hvna11.wav', 'hvna12.wav',
    'hvna2.wav', 'hvna3.wav', 'hvna4.wav', 'hvna5.wav', 'hvna6.wav', 'hvna7.wav' , 'hvna8.wav',
    'hvna9.wav', 'lvha1.wav', 'lvha10.wav', 'lvha11.wav', 'lvha12.wav', 'lvha2.wav', 'lvha3.wav',
    'lvha4.wav', 'lvha5.wav','lvha6.wav', 'lvha7.wav', 'lvha8.wav', 'lvha9.wav', 'lvla1.wav',
    'lvla10.wav', 'lvla11.wav', 'lvla12.wav', 'lvla2.wav', 'lvla3.wav', 'lvla4.wav', 'lvla5.wav',
    'lvla6.wav', 'lvla7.wav', 'lvla8.wav' , 'lvla9.wav', 'lvna1.wav', 'lvna10.wav', 'lvna11.wav',
    'lvna12.wav', 'lvna2.wav', 'lvna3.wav', 'lvna4.wav', 'lvna5.wav', 'lvna6.wav', 'lvna7.wav',
    'lvna8.wav', 'lvna9.wav', 'nvha1.wav', 'nvha10.wav', 'nvha11.wav', 'nvha12.wav', 'nvha2.wav',
    'nvha3.wav', 'nvha4.wav', 'nvha5.wav', 'nvha6.wav', 'nvha7.wav', 'nvha8.wav', 'nvha9.wav',
    'nvla1.wav', 'nvla10.wav', 'nvla11.wav', 'nvla12.wav', 'nvla2.wav', 'nvla3.wav', 'nvla4.wav',
    'nvla5.wav', 'nvla6.wav', 'nvla7.wav', 'nvla8.wav', 'nvla9.wav', 'nvna1.wav', 'nvna10.wav',
    'nvna11.wav', 'nvna12.wav', 'nvna2.wav', 'nvna3.wav', 'nvna4.wav', 'nvna5.wav', 'nvna6.wav', 
    'nvna7.wav', 'nvna8.wav', 'nvna9.wav'
]

# !!!!! These files are 19s long, not 21s !!!!!

@dataclass
class CalibrationMusicId(MusicID):
    """Music ID for calibration data."""
    number: int
    
    def to_filename(self) -> str:
        """Convert calibration music ID to filename."""
        return wav_filenames_ordered_calibration[self.number]
    
    def to_fragment(self) -> Tuple[Union[float, None], Union[float, None]]:
        """Convert calibration music ID to fragment (start, end) in seconds."""
        return (None, None)


@dataclass
class TrainingMusicId(MusicID):
    """Music ID for training data."""
    emotion_code_1: int
    emotion_code_2: int
    session: Union[int, str]
    half_id: int
    
    def to_filename(self) -> str:
        """Convert training music ID to filename."""
        return f"training_{self.emotion_code_1}_{self.emotion_code_2}_{self.session}"

    def to_fragment(self) -> Tuple[Union[float, None], Union[float, None]]:
        """Convert training music ID to fragment (start, end) in seconds."""
        if self.half_id == 0:
            return (0.0, 20.0)
        elif self.half_id == 1:
            return (20.0, None)
        else:
            raise ValueError("half_id must be 0 or 1")


@dataclass
class EEGTrial:
    """Data class containing music ID, raw EEG data, and emotion code."""
    music_id: MusicID  # Music identifier object
    raw_eeg: BaseRaw
    # emotion_code: int  # Integer code representing emotional state
