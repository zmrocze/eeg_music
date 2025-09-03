# Complete Python code to load NMED-T dataset preserving structure
import scipy.io
import numpy as np
from pathlib import Path

import mat73


def load_mat_file(filepath):
  """
  Load MATLAB file with fallback to mat73 for v7.3 files
  """
  try:
    # Try scipy.io.loadmat first (works for older MATLAB formats)
    return scipy.io.loadmat(filepath)
  except Exception:
    return mat73.loadmat(filepath)


class NMEDTLoader:
  """
  Loader for the Naturalistic Music EEG Dataset - Tempo (NMED-T)

  Dataset Structure:
  - 20 participants, 10 songs each
  - EEG: 125 electrodes × time samples × 20 participants
  - Sampling rate: 125 Hz (preprocessed), 1000 Hz (raw)
  - Songs range from 55.97 to 150 BPM
  """

  def __init__(self, dataset_path):
    """
    Initialize the NMED-T loader

    Parameters:
    -----------
    dataset_path : str or Path
        Path to the folder containing NMED-T .mat files
    """
    self.dataset_path = Path(dataset_path)
    self.song_info = {
      1: {
        "title": "First Fires",
        "artist": "Bonobo",
        "tempo": 55.97,
        "tempo_hz": 0.9328,
      },
      2: {
        "title": "Oino",
        "artist": "LA Priest",
        "tempo": 69.44,
        "tempo_hz": 1.1574,
      },
      3: {
        "title": "Tiptoes",
        "artist": "Daedelus",
        "tempo": 74.26,
        "tempo_hz": 1.2376,
      },
      4: {
        "title": "Careless Love",
        "artist": "Croquet Club",
        "tempo": 82.42,
        "tempo_hz": 1.3736,
      },
      5: {
        "title": "Lebanese Blonde",
        "artist": "Thievery Corporation",
        "tempo": 91.46,
        "tempo_hz": 1.5244,
      },
      6: {
        "title": "Canopée",
        "artist": "Polo & Pan",
        "tempo": 96.15,
        "tempo_hz": 1.6026,
      },
      7: {
        "title": "Doing Yoga",
        "artist": "Kazy Lambist",
        "tempo": 108.70,
        "tempo_hz": 1.8116,
      },
      8: {
        "title": "Until the Sun Needs to Rise",
        "artist": "Rüfüs du Sol",
        "tempo": 120.00,
        "tempo_hz": 2.0000,
      },
      9: {
        "title": "Silent Shout",
        "artist": "The Knife",
        "tempo": 128.21,
        "tempo_hz": 2.1368,
      },
      10: {
        "title": "The Last Thing You Should Do",
        "artist": "David Bowie",
        "tempo": 150.00,
        "tempo_hz": 2.5000,
      },
    }

  def load_cleaned_eeg_data(self, song_numbers=None):
    """
    Load preprocessed/cleaned EEG data for specified songs

    Parameters:
    -----------
    song_numbers : list or None
        List of song numbers (1-10) to load. If None, loads all songs.

    Returns:
    --------
    dict
        Dictionary with song numbers as keys, containing:
        - 'data': EEG data array (125 electrodes × time × 20 participants)
        - 'participants': List of participant IDs
        - 'sampling_rate': Sampling rate (125 Hz)
        - 'song_info': Song metadata
    """
    if song_numbers is None:
      song_numbers = list(range(1, 11))

    eeg_data = {}

    for song_num in song_numbers:
      # File naming convention: song2X_Imputed.mat where X is song number
      filename = f"song2{song_num}_Imputed.mat"
      filepath = self.dataset_path / filename

      if not filepath.exists():
        print(f"Warning: File {filename} not found. Skipping song {song_num}")
        continue

      try:
        mat_data = load_mat_file(filepath)

        # Extract the main data variables
        data_key = f"data2{song_num}"  # e.g., 'data21', 'data22', etc.
        subs_key = f"subs2{song_num}"  # e.g., 'subs21', 'subs22', etc.

        eeg_data[song_num] = {
          "data": mat_data[data_key],  # Shape: (125, time_samples, 20)
          "participants": [
            item[0] for item in mat_data[subs_key][0]
          ],  # List of participant IDs
          "sampling_rate": mat_data["fs"][0][0],  # 125 Hz
          "song_info": self.song_info[song_num],
          "shape_description": "(electrodes=125, time_samples, participants=20)",
        }

        print(
          f"Loaded song {song_num}: {self.song_info[song_num]['title']} "
          f"- Shape: {eeg_data[song_num]['data'].shape}"
        )

      except Exception as e:
        print(f"Error loading song {song_num}: {e}")

    return eeg_data

  def load_raw_eeg_data(self, participant_ids=None, recording_sessions=None):
    """
    Load raw EEG data for specified participants and sessions

    Parameters:
    -----------
    participant_ids : list or None
        List of participant IDs to load. If None, loads all available.
    recording_sessions : list or None
        List of recording sessions [1, 2]. If None, loads both sessions.

    Returns:
    --------
    dict
        Dictionary with (participant_id, session) tuples as keys, containing:
        - 'data': Raw EEG data (129 electrodes × time_samples)
        - 'events': Event triggers and timing information
        - 'sampling_rate': Sampling rate (1000 Hz)
    """
    if recording_sessions is None:
      recording_sessions = [1, 2]

    raw_data = {}

    # Get all available raw files
    raw_files = list(self.dataset_path.glob("*_*_raw.mat"))

    if participant_ids is None:
      # Extract all participant IDs from available files
      participant_ids = set()
      for file in raw_files:
        parts = file.stem.split("_")
        if len(parts) >= 2:
          participant_ids.add(int(parts[0]))
      participant_ids = sorted(list(participant_ids))

    for participant_id in participant_ids:
      for session in recording_sessions:
        filename = f"{participant_id:02d}_{session}_raw.mat"
        filepath = self.dataset_path / filename

        if not filepath.exists():
          print(f"Warning: File {filename} not found. Skipping.")
          continue

        try:
          mat_data = load_mat_file(filepath)

          raw_data[(participant_id, session)] = {
            "data": mat_data["X"],  # Shape: (129, time_samples)
            "events": mat_data["DIN 1"],  # Event triggers
            "sampling_rate": mat_data["fs"][0][0],  # 1000 Hz
            "participant_id": participant_id,
            "session": session,
            "shape_description": "(electrodes=129, time_samples) - includes vertex reference",
          }

          print(
            f"Loaded raw data for participant {participant_id}, session {session} "
            f"- Shape: {raw_data[(participant_id, session)]['data'].shape}"
          )

        except Exception as e:
          print(
            f"Error loading raw data for participant {participant_id}, session {session}: {e}"
          )

    return raw_data

  def load_behavioral_data(self):
    """
    Load behavioral ratings (familiarity and enjoyment)

    Returns:
    --------
    dict
        Dictionary containing:
        - 'ratings': 3D array (20 participants × 10 songs × 2 questions)
        - 'questions': ['familiarity', 'enjoyment']
        - 'participants': Number of participants
        - 'songs': Number of songs
    """
    filepath = self.dataset_path / "behavioralRatings.mat"

    if not filepath.exists():
      print("Warning: behavioralRatings.mat not found")
      return None

    try:
      mat_data = load_mat_file(filepath)

      return {
        "ratings": mat_data["behavioralRatings"],  # Shape: (20, 10, 2)
        "questions": ["familiarity", "enjoyment"],
        "participants": 20,
        "songs": 10,
        "shape_description": "(participants=20, songs=10, questions=2)",
        "rating_scale": "1-9 scale",
      }

    except Exception as e:
      print(f"Error loading behavioral data: {e}")
      return None

  def load_participant_info(self):
    """
    Load participant demographic information

    Returns:
    --------
    dict
        Dictionary containing participant demographics:
        - age, musical training years, weekly listening hours, participant IDs
    """
    filepath = self.dataset_path / "participantInfo.mat"

    if not filepath.exists():
      print("Warning: participantInfo.mat not found")
      return None

    try:
      mat_data = load_mat_file(filepath)
      participant_info = mat_data["participantInfo"]

      # Extract fields from the struct array
      demographics = {}
      field_names = participant_info.dtype.names

      for i, field in enumerate(field_names):
        if field == "id":
          demographics[field] = [item[0][0] for item in participant_info[field][0]]
        else:
          demographics[field] = [
            item[0][0] if item.size > 0 else np.nan
            for item in participant_info[field][0]
          ]

      return {
        "demographics": demographics,
        "fields": ["age", "nYearsTraining", "weeklyListening", "id"],
        "descriptions": {
          "age": "Age in years",
          "nYearsTraining": "Years of musical training",
          "weeklyListening": "Weekly music listening hours",
          "id": "Participant identifier",
        },
      }

    except Exception as e:
      print(f"Error loading participant info: {e}")
      return None

  def load_tapping_data(self):
    """
    Load tapping response data

    Returns:
    --------
    dict
        Dictionary containing tapping responses and song orders
    """
    filepath = self.dataset_path / "TapIt.zip"

    if not filepath.exists():
      print("Warning: TapIt.zip not found")
      return None

    try:
      # For now, provide structure information
      # In practice, you'd need to extract the zip file first
      print("Note: TapIt.zip contains:")
      print("- TapIt.mat: Aggregated tapping data")
      print("- Individual .txt files: Raw tapping responses")

      return {
        "note": "TapIt.zip needs to be extracted first",
        "contains": {
          "TapIt.mat": {
            "allTappedResponses": "Cell array (20×10) of tap times",
            "allSongOrders": "Matrix (20×10) of stimulus presentation order",
          },
          "individual_files": "PPP_SS.txt format for each participant and song",
        },
      }

    except Exception as e:
      print(f"Error accessing tapping data: {e}")
      return None

  def get_dataset_summary(self):
    """
    Get a complete summary of the dataset structure

    Returns:
    --------
    dict
        Comprehensive dataset information
    """
    return {
      "dataset_name": "Naturalistic Music EEG Dataset - Tempo (NMED-T)",
      "participants": 20,
      "songs": 10,
      "electrodes": 125,  # after preprocessing (124 + 1 vertex)
      "raw_electrodes": 129,  # raw data includes face electrodes
      "sampling_rates": {"preprocessed": 125, "raw": 1000},  # Hz  # Hz
      "data_types": {
        "cleaned_eeg": "song2X_Imputed.mat (X=1-10)",
        "raw_eeg": "PP_R_raw.mat (PP=participant, R=recording 1-2)",
        "behavioral": "behavioralRatings.mat",
        "demographics": "participantInfo.mat",
        "tapping": "TapIt.zip",
      },
      "array_shapes": {
        "cleaned_eeg": "(125 electrodes, time_samples, 20 participants)",
        "raw_eeg": "(129 electrodes, time_samples)",
        "behavioral": "(20 participants, 10 songs, 2 questions)",
        "demographics": "Struct array with participant info",
      },
      "song_info": self.song_info,
      "total_size": "39GB",
      "license": "Creative Commons CC-BY",
    }


# Example usage
def main(dataset_path="./datasets/nmed-t"):
  """
  Example of how to use the NMED-T loader
  """
  # Initialize the loader with path to your downloaded NMED-T data
  loader = NMEDTLoader(dataset_path)

  # Get dataset overview
  print("=== NMED-T Dataset Summary ===")
  summary = loader.get_dataset_summary()
  print(f"Dataset: {summary['dataset_name']}")
  print(f"Participants: {summary['participants']}")
  print(f"Songs: {summary['songs']}")
  print(f"Electrodes (cleaned): {summary['electrodes']}")
  print(f"Sampling rate (cleaned): {summary['sampling_rates']['preprocessed']} Hz")

  # Load cleaned EEG data for specific songs
  print("\n=== Loading Cleaned EEG Data ===")
  eeg_data = loader.load_cleaned_eeg_data(song_numbers=[1, 2, 3])

  for song_num, data in eeg_data.items():
    print(
      f"Song {song_num}: {data['song_info']['title']} by {data['song_info']['artist']}"
    )
    print(f"  Tempo: {data['song_info']['tempo']} BPM")
    print(f"  EEG shape: {data['data'].shape} - {data['shape_description']}")
    print(f"  Sampling rate: {data['sampling_rate']} Hz")

  # Load behavioral data
  print("\n=== Loading Behavioral Data ===")
  behavioral = loader.load_behavioral_data()
  if behavioral:
    print(f"Behavioral ratings shape: {behavioral['ratings'].shape}")
    print(f"Questions: {behavioral['questions']}")
    print(f"Rating scale: {behavioral['rating_scale']}")

  # Load participant demographics
  print("\n=== Loading Participant Demographics ===")
  demographics = loader.load_participant_info()
  if demographics:
    print("Available demographic fields:")
    for field, desc in demographics["descriptions"].items():
      print(f"  {field}: {desc}")

  # Load raw EEG data for specific participants
  print("\n=== Loading Raw EEG Data ===")
  raw_data = loader.load_raw_eeg_data(participant_ids=[2, 3], recording_sessions=[1])

  for key, data in raw_data.items():
    participant_id, session = key
    print(f"Participant {participant_id}, Session {session}:")
    print(f"  Raw EEG shape: {data['data'].shape} - {data['shape_description']}")
    print(f"  Sampling rate: {data['sampling_rate']} Hz")


if __name__ == "__main__":
  main()
