"""
BCMI (Brain-Computer Music Interface) Dataset Loaders

This module provides polymorphic loaders for various BCMI datasets from the 
BCMI-MIdAS project (Brain-Computer Music Interface for Monitoring and Inducing 
Affective States), handling different experimental paradigms while sharing 
common functionality.

The BCMI datasets are part of a comprehensive study investigating neural correlates 
of music-induced emotions and developing brain-computer interfaces for real-time 
affective music generation. All datasets follow BIDS format and use EEG recordings 
with standardized emotional state annotations.

Available Datasets:
    - bcmi-calibration: System calibration with 40s dual-target trials
    - bcmi-training: Training sessions with dual-affect targeting  
    - bcmi-testing: Online BCI testing with target-measure-change paradigm
    - bcmi-tempo: Tempo control via motor imagery
    - bcmi-scores: Movie score-induced emotion experiments
    - bcmi-fmri: Joint EEG-fMRI with classical music listening

Emotional State Encoding:
    All datasets use a standardized 9-point emotional state grid based on the 
    valence-arousal model:
    - Valence: Negative (4,5,6), Neutral (7,8,9), Positive (1,2,3)
    - Arousal: Low (2,5,8), Neutral (3,6,9), High (1,4,7)

Technical Specifications:
    - Sampling Rate: 1000 Hz
    - Channels: 37 EEG electrodes + auxiliary channels
    - Format: BIDS-compliant EDF files
    - Events: TSV format with emotion codes and timing markers

Usage Example:
    >>> # Load single dataset
    >>> from bcmi import create_bcmi_loader
    >>> loader = create_bcmi_loader('/path/to/bcmi-calibration')
    >>> data = loader.load_all_subjects(max_subjects=5)
    >>> loader.get_dataset_statistics()
    >>> 
    >>> # Load specific subjects and runs
    >>> subject_data = loader.load_subject_data('01', max_runs=3)
    >>> 
    >>> # Get trials for specific emotional states
    >>> happy_trials = loader.get_condition_trials(1)  # High valence, high arousal
    >>> calm_trials = loader.get_condition_trials(8)   # Neutral valence, low arousal
    >>> 
    >>> # Access raw EEG data and events
    >>> raw_eeg = loader.data['01']['no_session']['1']['raw']
    >>> events = loader.data['01']['no_session']['1']['processed_events']
    >>> 
    >>> # Load all BCMI datasets at once
    >>> from bcmi import load_all_bcmi_datasets
    >>> all_loaders = load_all_bcmi_datasets('/path/to/bcmi', max_subjects_per_dataset=3)
    >>> 
    >>> # Compare across datasets
    >>> for name, loader in all_loaders.items():
    >>>     print(f"{name}: {len(loader.data)} subjects")
    >>>     loader.get_dataset_statistics()

References:
    Daly, I., Nicolaou, N., Williams, D., Hwang, F., Kirke, A., Miranda, E., & 
    Nasuto, S. J. (2020). Neural and physiological data from participants listening 
    to affective music. Scientific Data, 7(1), 177.
    
    Williams, D., Kirke, A., Miranda, E. R., Daly, I., Hwang, F., Weaver, J., & 
    Nasuto, S. J. (2017). Affective calibration of musical feature sets in an 
    emotionally intelligent music composition system. ACM Transactions on Applied 
    Perception, 14(3), 1-13.
"""

from abc import ABC, abstractmethod
from mne_bids import get_entity_vals, BIDSPath, read_raw_bids
import pandas as pd
import os
from pathlib import Path
from typing import Dict, List, Optional, Union, Any


class BaseBCMILoader(ABC):
    """
    Abstract base class for BCMI dataset loaders.
    
    Provides common functionality for loading BIDS-formatted EEG data with emotional 
    state annotations following the valence-arousal grid. This class implements the 
    shared loading logic while allowing specialized classes to handle dataset-specific 
    experimental paradigms.
    
    The base loader handles:
    - BIDS path construction and EEG data loading
    - Common event processing (emotion codes vs. timing markers)
    - Subject/session/run management
    - Standardized emotional state mapping
    - Dataset statistics and querying
    
    Attributes:
        root_path (Path): Path to the dataset root directory
        dataset_name (str): Name of the dataset (extracted from path)
        subjects (List[str]): Available subject IDs
        data (Dict): Loaded EEG data organized by subject/session/run
        emotional_states (Dict): 9-point emotion grid mapping
    
    Usage:
        This is an abstract class. Use concrete implementations like:
        >>> loader = BCMICalibrationLoader('/path/to/bcmi-calibration')
        >>> data = loader.load_all_subjects()
        
        Or use the factory function:
        >>> loader = create_bcmi_loader('/path/to/bcmi-calibration')
    """
    
    def __init__(self, root_path: str):
        """
        Initialize the BCMI loader.
        
        Args:
            root_path: Path to the BCMI dataset root directory
        """
        self.root_path = Path(root_path)
        self.dataset_name = self.root_path.name
        self.subjects = self._get_available_subjects()
        self.data = {}
        
        # Standard 9-point emotional state mapping (3x3 valence x arousal)
        self.emotional_states = {
            1: {'valence': 'High', 'arousal': 'High', 'description': 'Excited/Happy'},
            2: {'valence': 'High', 'arousal': 'Low', 'description': 'Peaceful/Content'}, 
            3: {'valence': 'High', 'arousal': 'Neutral', 'description': 'Positive/Pleasant'},
            4: {'valence': 'Low', 'arousal': 'High', 'description': 'Angry/Agitated'},
            5: {'valence': 'Low', 'arousal': 'Low', 'description': 'Sad/Depressed'},
            6: {'valence': 'Low', 'arousal': 'Neutral', 'description': 'Negative/Unpleasant'},
            7: {'valence': 'Neutral', 'arousal': 'High', 'description': 'Alert/Activated'},
            8: {'valence': 'Neutral', 'arousal': 'Low', 'description': 'Calm/Relaxed'},
            9: {'valence': 'Neutral', 'arousal': 'Neutral', 'description': 'Neutral/Balanced'}
        }
    
    def _get_available_subjects(self) -> List[str]:
        """Get list of available subjects in the dataset."""
        try:
            return get_entity_vals(str(self.root_path), 'subject')
        except Exception:
            # Fallback: scan for subject directories
            subject_dirs = [d.name for d in self.root_path.iterdir() 
                          if d.is_dir() and d.name.startswith('sub-')]
            return [s.replace('sub-', '') for s in subject_dirs]
    
    def _get_available_sessions(self, subject_id: str) -> List[str]:
        """Get available sessions for a subject (returns [''] if no sessions)."""
        try:
            # Check if session directories exist directly
            subject_path = self.root_path / f'sub-{subject_id}'
            if not subject_path.exists():
                return ['']
            
            session_dirs = [d.name for d in subject_path.iterdir() 
                          if d.is_dir() and d.name.startswith('ses-')]
            if session_dirs:
                return [s.replace('ses-', '') for s in session_dirs]
            return ['']
        except Exception:
            return ['']
    
    def _get_available_runs(self, subject_id: str, session: str = '') -> List[str]:
        """Get available runs for a subject/session."""
        try:
            # Try to get task names using mne_bids
            tasks = get_entity_vals(str(self.root_path), 'task')
            runs = [task.replace('run', '') for task in tasks if task.startswith('run')]
            return sorted(runs) if runs else ['1']
        except Exception:
            return ['1', '2', '3', '4', '5']  # Common fallback
    
    @abstractmethod
    def _get_experimental_info(self) -> Dict[str, Any]:
        """Get dataset-specific experimental information."""
        pass
    
    @abstractmethod
    def _process_events_specific(self, events_df: pd.DataFrame, 
                               run_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process events according to dataset-specific experimental paradigm."""
        pass
    
    def _process_events_common(self, events_df: pd.DataFrame) -> Dict[str, Any]:
        """Common event processing for all BCMI datasets."""
        try:
            # Separate emotion codes (1-9) from markers (100+)
            emotion_events = events_df[events_df['trial_type'] <= 9].copy()
            marker_events = events_df[events_df['trial_type'] > 100].copy()
            
            # Add emotional state information
            if not emotion_events.empty:
                emotion_events['emotion_description'] = emotion_events['trial_type'].map(
                    lambda x: self.emotional_states.get(x, {}).get('description', 'Unknown')
                )
                emotion_events['valence'] = emotion_events['trial_type'].map(
                    lambda x: self.emotional_states.get(x, {}).get('valence', 'Unknown')
                )
                emotion_events['arousal'] = emotion_events['trial_type'].map(
                    lambda x: self.emotional_states.get(x, {}).get('arousal', 'Unknown')
                )
            
            return {
                'emotion_events': emotion_events,
                'marker_events': marker_events,
                'all_events': events_df
            }
        except Exception as e:
            print(f"    Warning: Could not process events - {str(e)[:50]}")
            return {'all_events': events_df}
    
    def load_subject_data(self, subject_id: str, max_runs: Optional[int] = None, 
                         max_sessions: Optional[int] = None) -> Dict[str, Any]:
        """
        Load data for one subject across all sessions and runs.
        
        This method loads EEG data, events, and metadata for a single subject.
        It handles both single-session and multi-session datasets automatically.
        
        Args:
            subject_id: Subject identifier (without 'sub-' prefix, e.g., '01', '02')
            max_runs: Maximum number of runs per session to load (None = all)
            max_sessions: Maximum number of sessions to load (None = all)
            
        Returns:
            Dictionary containing all loaded data for the subject, structured as:
            {
                'session_key': {
                    'run_id': {
                        'raw': MNE Raw object,
                        'events': pandas DataFrame,
                        'processed_events': Dict with emotion/marker events,
                        'duration': float (seconds),
                        'n_channels': int,
                        'sfreq': float (Hz),
                        'n_trials': int,
                        'experimental_info': Dict
                    }
                }
            }
            
        Example:
            >>> loader = BCMICalibrationLoader('/path/to/bcmi-calibration')
            >>> subject_data = loader.load_subject_data('01', max_runs=3)
            >>> 
            >>> # Access raw EEG data
            >>> raw = subject_data['no_session']['1']['raw']
            >>> print(f"Channels: {raw.ch_names}")
            >>> print(f"Duration: {raw.times[-1]:.1f} seconds")
            >>> 
            >>> # Access emotion events
            >>> events = subject_data['no_session']['1']['processed_events']['emotion_events']
            >>> print(f"Emotion conditions: {events['trial_type'].unique()}")
        """
        self.data[subject_id] = {}
        sessions = self._get_available_sessions(subject_id)
        
        if max_sessions:
            sessions = sessions[:max_sessions]
        
        print(f"Loading subject {subject_id} ({self.dataset_name}):")
        
        for session in sessions:
            session_key = session if session else 'no_session'
            self.data[subject_id][session_key] = {}
            
            runs = self._get_available_runs(subject_id, session)
            if max_runs:
                runs = runs[:max_runs]
            
            for run in runs:
                try:
                    # Construct BIDS path - handle both run numbers and task names
                    if run.isdigit():
                        # Traditional run number (e.g., '1', '2', '3')
                        task_name = f'run{run}'
                    else:
                        # Direct task name (e.g., 'classicalMusic', 'genMusic01')
                        task_name = run
                    
                    if session:
                        bids_path = BIDSPath(
                            subject=subject_id,
                            session=session,
                            task=task_name,
                            datatype='eeg',
                            root=str(self.root_path)
                        )
                    else:
                        bids_path = BIDSPath(
                            subject=subject_id,
                            task=task_name,
                            datatype='eeg',
                            root=str(self.root_path)
                        )
                    
                    # Load EEG data
                    raw = read_raw_bids(bids_path, verbose=False)
                    
                    # Load events
                    events_path = bids_path.copy().update(suffix='events', extension='.tsv')
                    events_df = pd.read_csv(events_path.fpath, sep='\t')
                    
                    # Process events (common + specific)
                    common_events = self._process_events_common(events_df)
                    
                    # Store basic run data
                    run_data = {
                        'raw': raw,
                        'events': events_df,
                        'processed_events': common_events,
                        'duration': raw.duration,
                        'n_channels': raw.info['nchan'],
                        'sfreq': raw.info['sfreq'],
                        'experimental_info': self._get_experimental_info()
                    }
                    
                    # Add dataset-specific processing
                    specific_events = self._process_events_specific(events_df, run_data)
                    run_data['processed_events'].update(specific_events)
                    
                    # Calculate trial count
                    emotion_events = run_data['processed_events'].get('emotion_events', pd.DataFrame())
                    run_data['n_trials'] = len(emotion_events)
                    
                    self.data[subject_id][session_key][run] = run_data
                    
                    session_str = f" ses-{session}" if session else ""
                    print(f"  ✓ Run {run}{session_str}: {raw.times[-1]:.1f}s, "
                          f"{len(events_df)} events, {run_data['n_trials']} trials")
                    
                except Exception as e:
                    session_str = f" ses-{session}" if session else ""
                    print(f"  ✗ Run {run}{session_str}: {str(e)[:50]}...")
        
        return self.data[subject_id]
    
    def load_all_subjects(self, max_subjects: Optional[int] = None, 
                         max_runs_per_session: Optional[int] = None,
                         max_sessions_per_subject: Optional[int] = None,
                         verbose: bool = True) -> Dict[str, Any]:
        """
        Load data for all available subjects.
        
        This method provides bulk loading of the entire dataset with optional 
        limits for memory management. It includes progress tracking and error 
        handling for robust loading of large datasets.
        
        Args:
            max_subjects: Maximum number of subjects to load (None = all)
            max_runs_per_session: Maximum runs per session (None = all)
            max_sessions_per_subject: Maximum sessions per subject (None = all)
            verbose: Whether to print detailed loading progress
            
        Returns:
            Dictionary containing all loaded data, structured as:
            {
                'subject_id': {
                    'session_key': {
                        'run_id': {
                            'raw': MNE Raw object,
                            'events': pandas DataFrame,
                            'processed_events': Dict,
                            ...
                        }
                    }
                }
            }
            
        Example:
            >>> loader = BCMICalibrationLoader('/path/to/bcmi-calibration')
            >>> 
            >>> # Load all subjects (memory intensive)
            >>> all_data = loader.load_all_subjects()
            >>> 
            >>> # Load subset for initial exploration
            >>> sample_data = loader.load_all_subjects(
            ...     max_subjects=5, 
            ...     max_runs_per_session=2
            ... )
            >>> 
            >>> # Access loaded data
            >>> for subject_id, subject_data in sample_data.items():
            ...     for session_key, session_data in subject_data.items():
            ...         for run_id, run_data in session_data.items():
            ...             print(f"Subject {subject_id}, Run {run_id}: "
            ...                   f"{run_data['n_trials']} trials")
        """
        subjects_to_load = self.subjects[:max_subjects] if max_subjects else self.subjects
        
        if verbose:
            print(f"🔄 Loading {self.dataset_name} dataset...")
            print(f"📦 Subjects to load: {len(subjects_to_load)} of {len(self.subjects)}")
            if max_runs_per_session:
                print(f"🏃 Max runs per session: {max_runs_per_session}")
            if max_sessions_per_subject:
                print(f"📅 Max sessions per subject: {max_sessions_per_subject}")
            print()
        
        successful_subjects = 0
        failed_subjects = 0
        
        for subject in subjects_to_load:
            try:
                self.load_subject_data(subject, max_runs_per_session, max_sessions_per_subject)
                if subject in self.data and self.data[subject]:
                    successful_subjects += 1
                else:
                    failed_subjects += 1
            except Exception as e:
                if verbose:
                    print(f"✗ Subject {subject}: {str(e)[:50]}...")
                failed_subjects += 1
        
        if verbose:
            print(f"\n📊 LOADING SUMMARY:")
            print(f"• Successfully loaded: {successful_subjects} subjects")
            print(f"• Failed to load: {failed_subjects} subjects")
            if successful_subjects + failed_subjects > 0:
                print(f"• Success rate: {successful_subjects/(successful_subjects+failed_subjects)*100:.1f}%")
        
        return self.data
    
    def get_dataset_statistics(self) -> None:
        """Print comprehensive dataset statistics."""
        if not self.data:
            print("No data loaded. Call load_all_subjects() or load_subject_data() first.")
            return
        
        total_sessions = 0
        total_runs = 0
        total_trials = 0
        all_conditions = []
        
        print(f"📈 {self.dataset_name.upper()} DATASET STATISTICS:")
        print(f"• Loaded subjects: {len(self.data)}")
        
        for subject_id, subject_data in self.data.items():
            for session_key, session_data in subject_data.items():
                total_sessions += 1
                total_runs += len(session_data)
                
                for run_data in session_data.values():
                    total_trials += run_data.get('n_trials', 0)
                    
                    # Collect emotion conditions
                    processed_events = run_data.get('processed_events', {})
                    emotion_events = processed_events.get('emotion_events', pd.DataFrame())
                    if not emotion_events.empty:
                        conditions = emotion_events['trial_type'].tolist()
                        all_conditions.extend(conditions)
        
        print(f"• Total sessions: {total_sessions}")
        print(f"• Total runs: {total_runs}")
        print(f"• Total trials: {total_trials}")
        
        # Experimental info
        exp_info = self._get_experimental_info()
        print(f"\n🎵 EXPERIMENTAL PARADIGM:")
        print(f"• Type: {exp_info.get('paradigm_type', 'Unknown')}")
        print(f"• Trial structure: {exp_info.get('trial_structure', 'Unknown')}")
        print(f"• Music type: {exp_info.get('music_type', 'Unknown')}")
        
        if all_conditions:
            condition_counts = pd.Series(all_conditions).value_counts().sort_index()
            print(f"\n🎭 EMOTIONAL CONDITION DISTRIBUTION:")
            for condition, count in condition_counts.items():
                if isinstance(condition, int) and condition in self.emotional_states:
                    desc = self.emotional_states[condition].get('description', 'Unknown')
                    print(f"  Condition {condition} ({desc}): {count:3d} trials")
                else:
                    print(f"  Condition {condition}: {count:3d} trials")
        
        # Technical specifications
        if self.data:
            # Get sample from first available data
            sample_subject = list(self.data.keys())[0]
            sample_session = list(self.data[sample_subject].keys())[0]
            sample_run = list(self.data[sample_subject][sample_session].keys())[0]
            sample_data = self.data[sample_subject][sample_session][sample_run]
            
            print(f"\n⚙️  TECHNICAL SPECIFICATIONS:")
            print(f"• Sampling rate: {sample_data['sfreq']} Hz")
            print(f"• Channels per recording: {sample_data['n_channels']}")
            print(f"• Average run duration: ~{sample_data['duration']:.0f} seconds")
    
    def get_condition_trials(self, condition_code: int) -> List[Dict[str, Any]]:
        """
        Get all trials for a specific emotional condition across all subjects.
        
        This method extracts trials matching a specific emotion code from all 
        loaded subjects, sessions, and runs. Useful for emotion-specific analysis
        and cross-subject comparisons.
        
        Args:
            condition_code: Emotional condition code (1-9) from the valence-arousal grid:
                1: High valence, High arousal (Excited/Happy)
                2: High valence, Low arousal (Peaceful/Content)
                3: High valence, Neutral arousal (Positive/Pleasant)
                4: Low valence, High arousal (Angry/Agitated)
                5: Low valence, Low arousal (Sad/Depressed)
                6: Low valence, Neutral arousal (Negative/Unpleasant)
                7: Neutral valence, High arousal (Alert/Activated)
                8: Neutral valence, Low arousal (Calm/Relaxed)
                9: Neutral valence, Neutral arousal (Neutral/Balanced)
            
        Returns:
            List of trial dictionaries, each containing:
            {
                'subject': str (subject ID),
                'session': str or None (session ID),
                'run': str (run ID),
                'trial_info': pandas Series (trial details),
                'raw_data': MNE Raw object,
                'dataset': str (dataset name)
            }
            
        Example:
            >>> loader = BCMICalibrationLoader('/path/to/bcmi-calibration')
            >>> loader.load_all_subjects(max_subjects=3)
            >>> 
            >>> # Get all happy/excited trials (high valence, high arousal)
            >>> happy_trials = loader.get_condition_trials(1)
            >>> print(f"Found {len(happy_trials)} happy trials")
            >>> 
            >>> # Analyze trial distribution across subjects
            >>> for trial in happy_trials:
            ...     print(f"Subject {trial['subject']}, Run {trial['run']}: "
            ...           f"{trial['trial_info']['emotion_description']}")
            >>> 
            >>> # Extract EEG data for a specific trial
            >>> trial = happy_trials[0]
            >>> raw_eeg = trial['raw_data']
            >>> onset = trial['trial_info']['onset']
            >>> duration = trial['trial_info']['duration']
            >>> trial_data = raw_eeg.copy().crop(onset, onset + duration)
        """
        condition_trials = []
        
        for subject_id, subject_data in self.data.items():
            for session_key, session_data in subject_data.items():
                for run_id, run_data in session_data.items():
                    processed_events = run_data.get('processed_events', {})
                    emotion_events = processed_events.get('emotion_events', pd.DataFrame())
                    
                    if not emotion_events.empty:
                        matching_trials = emotion_events[emotion_events['trial_type'] == condition_code]
                        
                        for _, trial in matching_trials.iterrows():
                            condition_trials.append({
                                'subject': subject_id,
                                'session': session_key if session_key != 'no_session' else None,
                                'run': run_id,
                                'trial_info': trial,
                                'raw_data': run_data['raw'],
                                'dataset': self.dataset_name
                            })
        
        return condition_trials
    
    def get_emotional_state_info(self, condition_code: int) -> Dict[str, str]:
        """Get information about an emotional state."""
        return self.emotional_states.get(condition_code, {})
    
    def get_available_subjects(self) -> List[str]:
        """Get list of all available subjects."""
        return self.subjects


class BCMICalibrationLoader(BaseBCMILoader):
    """
    Loader for BCMI Calibration dataset.
    
    The calibration dataset contains system calibration data where participants 
    listened to 40-second music clips with dual-target affective states. Each trial 
    consists of two consecutive 20-second segments, each targeting a different 
    emotional state from the 9-point valence-arousal grid.
    
    Experimental Design:
        - Purpose: Calibrate BCI system for individual differences
        - Participants: 19 healthy adults
        - Sessions: 1 per subject
        - Runs: 5 runs per session
        - Trials: 18 trials per run (90 total per subject)
        - Trial Structure: 40s total (20s state1 + 20s state2)
        - Music: Synthetic, real-time generated
        - Sampling Rate: 1000 Hz
        - Channels: 37 EEG + auxiliary
    
    Data Structure:
        Each trial contains paired emotion targets, making this dataset ideal for 
        studying emotion transitions and individual calibration patterns.
    
    Usage:
        >>> loader = BCMICalibrationLoader('/path/to/bcmi-calibration')
        >>> data = loader.load_all_subjects()
        >>> 
        >>> # Access calibration-specific trial structure
        >>> subject_data = loader.load_subject_data('01')
        >>> run_data = subject_data['no_session']['1']
        >>> calibration_trials = run_data['processed_events']['calibration_trials']
        >>> 
        >>> # Examine dual-state trials
        >>> for trial in calibration_trials[:3]:
        ...     state1 = trial['state_1']['description'] 
        ...     state2 = trial['state_2']['description']
        ...     print(f"Trial {trial['trial_number']}: {state1} → {state2}")
        >>> 
        >>> # Get transition patterns
        >>> transitions = [(t['state_1']['code'], t['state_2']['code']) 
        ...               for t in calibration_trials]
        >>> print(f"Emotion transitions: {set(transitions)}")
    
    References:
        Daly, I., Williams, D., Hwang, F., Kirke, A., Malik, A., Roesch, E., ... & 
        Nasuto, S. J. (2015). Identifying music-induced emotions from EEG for use 
        in brain-computer music interfacing. In 4th Workshop on Affective 
        Brain-Computer Interfaces.
    """
    
    def _get_experimental_info(self) -> Dict[str, Any]:
        return {
            'paradigm_type': 'Calibration',
            'trial_structure': '40s total (20s state1 + 20s state2)',
            'music_type': 'Synthetic real-time generated',
            'runs_per_subject': 5,
            'trials_per_run': 18,
            'description': 'System calibration with dual-target affective trials'
        }
    
    def _process_events_specific(self, events_df: pd.DataFrame, 
                               run_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process calibration-specific events with dual-state trials."""
        try:
            emotion_events = run_data['processed_events'].get('emotion_events', pd.DataFrame())
            if emotion_events.empty:
                return {}
            
            # Group events into trial pairs (each trial has two emotional targets)
            trials = []
            for i in range(0, len(emotion_events), 2):
                if i + 1 < len(emotion_events):
                    trial_1 = emotion_events.iloc[i]
                    trial_2 = emotion_events.iloc[i + 1]
                    
                    trials.append({
                        'trial_number': i // 2 + 1,
                        'onset': trial_1['onset'],
                        'duration': trial_1['duration'] + trial_2['duration'],
                        'state_1': {
                            'code': trial_1['trial_type'],
                            'onset': trial_1['onset'],
                            'duration': trial_1['duration'],
                            'description': trial_1['emotion_description']
                        },
                        'state_2': {
                            'code': trial_2['trial_type'],
                            'onset': trial_2['onset'],
                            'duration': trial_2['duration'],
                            'description': trial_2['emotion_description']
                        }
                    })
            
            return {'calibration_trials': trials}
            
        except Exception as e:
            print(f"    Warning: Could not process calibration events - {str(e)[:50]}")
            return {}


class BCMITrainingLoader(BaseBCMILoader):
    """
    Loader for BCMI Training dataset.
    
    The training dataset contains multi-session training data where participants 
    learned to control a brain-computer music interface. Each trial consists of 
    20s+20s segments targeting different emotional states, allowing the system 
    to learn individual neural patterns for emotion discrimination.
    
    Experimental Design:
        - Purpose: Train BCI system on individual neural signatures
        - Participants: Variable (subset of calibration participants)
        - Sessions: Multiple training sessions per subject
        - Trial Structure: 40s total (20s + 20s targeting different affects)
        - Music: Synthetic generated based on training feedback
        - Focus: Learning dual-affective state contrasts
    
    Data Structure:
        Each trial contains consecutive pairs targeting different emotional states,
        optimized for training discriminative models between emotion categories.
    
    Usage:
        >>> loader = BCMITrainingLoader('/path/to/bcmi-training')
        >>> data = loader.load_all_subjects()
        >>> 
        >>> # Access training-specific trial pairs
        >>> subject_data = loader.load_subject_data('08')  # Example subject
        >>> session_data = subject_data['3']  # Session 3
        >>> run_data = session_data['1']  # Run 1
        >>> training_pairs = run_data['processed_events']['training_pairs']
        >>> 
        >>> # Analyze affect contrasts used in training
        >>> for pair in training_pairs[:5]:
        ...     print(f"Pair {pair['pair_number']}: {pair['contrast']}")
        >>> 
        >>> # Extract learning progression across sessions
        >>> all_contrasts = []
        >>> for session_key, session_data in subject_data.items():
        ...     for run_data in session_data.values():
        ...         pairs = run_data['processed_events'].get('training_pairs', [])
        ...         contrasts = [p['contrast'] for p in pairs]
        ...         all_contrasts.extend(contrasts)
        >>> print(f"Training contrasts: {set(all_contrasts)}")
    """
    
    def _get_experimental_info(self) -> Dict[str, Any]:
        return {
            'paradigm_type': 'Training',
            'trial_structure': '40s total (20s + 20s targeting different affects)',
            'music_type': 'Synthetic generated',
            'sessions_per_subject': 'Variable',
            'description': 'Training sessions for dual-affective state targeting'
        }
    
    def _process_events_specific(self, events_df: pd.DataFrame, 
                               run_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process training-specific events."""
        try:
            emotion_events = run_data['processed_events'].get('emotion_events', pd.DataFrame())
            if emotion_events.empty:
                return {}
            
            # Training trials are consecutive pairs targeting different affects
            training_pairs = []
            for i in range(0, len(emotion_events), 2):
                if i + 1 < len(emotion_events):
                    affect_1 = emotion_events.iloc[i]
                    affect_2 = emotion_events.iloc[i + 1]
                    
                    training_pairs.append({
                        'pair_number': i // 2 + 1,
                        'affect_1': {
                            'code': affect_1['trial_type'],
                            'onset': affect_1['onset'],
                            'description': affect_1['emotion_description']
                        },
                        'affect_2': {
                            'code': affect_2['trial_type'],
                            'onset': affect_2['onset'],
                            'description': affect_2['emotion_description']
                        },
                        'contrast': f"{affect_1['emotion_description']} → {affect_2['emotion_description']}"
                    })
            
            return {'training_pairs': training_pairs}
            
        except Exception as e:
            print(f"    Warning: Could not process training events - {str(e)[:50]}")
            return {}


class BCMITestingLoader(BaseBCMILoader):
    """
    Loader for BCMI Testing dataset.
    
    Experimental paradigm: Online testing with 20s+20s+20s structure:
    - Target affection phase
    - Measure affection phase  
    - Attempt to change affection phase
    """
    
    def _get_experimental_info(self) -> Dict[str, Any]:
        return {
            'paradigm_type': 'Testing',
            'trial_structure': '60s total (20s target + 20s measure + 20s change)',
            'music_type': 'Real-time generated (online BCI)',
            'description': 'Online testing with target-measure-change paradigm'
        }
    
    def _process_events_specific(self, events_df: pd.DataFrame, 
                               run_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process testing-specific events with target-measure-change structure."""
        try:
            emotion_events = run_data['processed_events'].get('emotion_events', pd.DataFrame())
            if emotion_events.empty:
                return {}
            
            # Testing trials have target-measure-change triplets
            testing_trials = []
            for i in range(0, len(emotion_events), 3):
                if i + 2 < len(emotion_events):
                    target = emotion_events.iloc[i]
                    measure = emotion_events.iloc[i + 1]
                    change = emotion_events.iloc[i + 2]
                    
                    testing_trials.append({
                        'trial_number': i // 3 + 1,
                        'target_phase': {
                            'code': target['trial_type'],
                            'onset': target['onset'],
                            'description': target['emotion_description']
                        },
                        'measure_phase': {
                            'code': measure['trial_type'],
                            'onset': measure['onset'],
                            'description': measure['emotion_description']
                        },
                        'change_phase': {
                            'code': change['trial_type'],
                            'onset': change['onset'],
                            'description': change['emotion_description']
                        },
                        'bci_sequence': f"Target: {target['emotion_description']} → "
                                      f"Measure: {measure['emotion_description']} → "
                                      f"Change: {change['emotion_description']}"
                    })
            
            return {'testing_trials': testing_trials}
            
        except Exception as e:
            print(f"    Warning: Could not process testing events - {str(e)[:50]}")
            return {}


class BCMITempoLoader(BaseBCMILoader):
    """
    Loader for BCMI Tempo dataset.
    
    Experimental paradigm: Tempo-based BCI where participants control 
    music tempo through imagined movement.
    """
    
    def _get_experimental_info(self) -> Dict[str, Any]:
        return {
            'paradigm_type': 'Tempo Control',
            'trial_structure': 'Variable duration tempo control trials',
            'music_type': 'Synthetic with real-time tempo modulation',
            'control_method': 'Imagined movement → tempo changes',
            'description': 'BCI for tempo control via motor imagery'
        }
    
    def _process_events_specific(self, events_df: pd.DataFrame, 
                               run_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process tempo-control specific events."""
        try:
            emotion_events = run_data['processed_events'].get('emotion_events', pd.DataFrame())
            if emotion_events.empty:
                return {}
            
            # Tempo trials may have different structure
            tempo_trials = []
            for idx, event in emotion_events.iterrows():
                tempo_trials.append({
                    'trial_number': idx + 1,
                    'emotion_target': {
                        'code': event['trial_type'],
                        'onset': event['onset'],
                        'duration': event['duration'],
                        'description': event['emotion_description']
                    },
                    'control_task': 'Tempo modulation via motor imagery'
                })
            
            return {'tempo_trials': tempo_trials}
            
        except Exception as e:
            print(f"    Warning: Could not process tempo events - {str(e)[:50]}")
            return {}


class BCMIScoresLoader(BaseBCMILoader):
    """
    Loader for BCMI Scores dataset.
    
    Experimental paradigm: Listening to movie scores for emotion induction.
    """
    
    def _get_experimental_info(self) -> Dict[str, Any]:
        return {
            'paradigm_type': 'Movie Scores',
            'trial_structure': 'Music listening trials',
            'music_type': 'Movie score excerpts',
            'description': 'Emotion induction via movie score listening'
        }
    
    def _process_events_specific(self, events_df: pd.DataFrame, 
                               run_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process movie scores specific events."""
        try:
            emotion_events = run_data['processed_events'].get('emotion_events', pd.DataFrame())
            if emotion_events.empty:
                return {}
            
            # Movie score trials
            score_trials = []
            for idx, event in emotion_events.iterrows():
                score_trials.append({
                    'trial_number': idx + 1,
                    'music_condition': {
                        'code': event['trial_type'],
                        'onset': event['onset'],
                        'duration': event['duration'],
                        'target_emotion': event['emotion_description']
                    },
                    'stimulus_type': 'Movie score excerpt'
                })
            
            return {'score_trials': score_trials}
            
        except Exception as e:
            print(f"    Warning: Could not process scores events - {str(e)[:50]}")
            return {}


class BCMIFMRILoader(BaseBCMILoader):
    """
    Loader for BCMI fMRI dataset.
    
    Experimental paradigm: Joint EEG-fMRI during classical music listening
    with emotion reporting via FEELTRACE interface.
    
    This dataset has a different structure from other BCMI datasets:
    - Uses task names instead of run numbers
    - Tasks: classicalMusic, genMusic01, genMusic02, genMusic03, washout
    - Subject format: sub-01, sub-02, etc. (not sub-001)
    """
    
    def _get_available_runs(self, subject_id: str, session: str = '') -> List[str]:
        """Get available task names for BCMI-fMRI dataset."""
        try:
            # BCMI-fMRI uses specific task names instead of run numbers
            return ['classicalMusic', 'genMusic01', 'genMusic02', 'genMusic03', 'washout']
        except Exception:
            return ['classicalMusic', 'genMusic01', 'genMusic02', 'genMusic03', 'washout']
    
    def _get_experimental_info(self) -> Dict[str, Any]:
        return {
            'paradigm_type': 'EEG-fMRI',
            'trial_structure': '5 tasks: classicalMusic + 3 genMusic runs + washout',
            'music_type': 'Generated music + classical pieces',
            'additional_data': 'Simultaneous fMRI acquisition',
            'emotion_reporting': 'FEELTRACE interface',
            'tasks': ['classicalMusic', 'genMusic01', 'genMusic02', 'genMusic03', 'washout'],
            'description': 'Joint EEG-fMRI with classical music and continuous emotion reporting'
        }
    
    def _process_events_specific(self, events_df: pd.DataFrame, 
                               run_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process fMRI-specific events with classical music trials."""
        try:
            emotion_events = run_data['processed_events'].get('emotion_events', pd.DataFrame())
            if emotion_events.empty:
                return {}
            
            # fMRI trials with classical music
            fmri_trials = []
            for idx, event in emotion_events.iterrows():
                fmri_trials.append({
                    'trial_number': idx + 1,
                    'music_condition': {
                        'code': event['trial_type'],
                        'onset': event['onset'],
                        'duration': event['duration'],
                        'target_emotion': event['emotion_description']
                    },
                    'recording_method': 'Simultaneous EEG-fMRI',
                    'emotion_reporting': 'FEELTRACE continuous'
                })
            
            return {'fmri_trials': fmri_trials}
            
        except Exception as e:
            print(f"    Warning: Could not process fMRI events - {str(e)[:50]}")
            return {}


def create_bcmi_loader(dataset_path: str) -> BaseBCMILoader:
    """
    Factory function to create the appropriate BCMI loader based on dataset path.
    
    This function automatically detects the dataset type from the directory name
    and returns the corresponding specialized loader. It provides a unified 
    interface for loading any BCMI dataset without needing to know the specific
    loader class.
    
    Args:
        dataset_path: Path to the BCMI dataset directory. The directory name
                     should follow the pattern 'bcmi-{type}' where type is one of:
                     calibration, training, testing, tempo, scores, fmri
        
    Returns:
        Appropriate BCMI loader instance for the detected dataset type.
        Falls back to BCMICalibrationLoader if type cannot be determined.
        
    Raises:
        FileNotFoundError: If the dataset path does not exist
        PermissionError: If the dataset path is not accessible
        
    Example:
        >>> # Automatic detection and loading
        >>> loader = create_bcmi_loader('/path/to/bcmi-calibration')
        >>> print(type(loader).__name__)  # BCMICalibrationLoader
        >>> 
        >>> # Load and explore different dataset types
        >>> datasets = [
        ...     '/path/to/bcmi-calibration',
        ...     '/path/to/bcmi-training', 
        ...     '/path/to/bcmi-testing'
        ... ]
        >>> 
        >>> for dataset_path in datasets:
        ...     loader = create_bcmi_loader(dataset_path)
        ...     data = loader.load_all_subjects(max_subjects=2)
        ...     print(f"{loader.dataset_name}: {len(loader.data)} subjects")
        ...     exp_info = loader._get_experimental_info()
        ...     print(f"  Paradigm: {exp_info['paradigm_type']}")
        ...     print(f"  Structure: {exp_info['trial_structure']}")
        
    Supported Dataset Types:
        - bcmi-calibration: System calibration with dual-target trials
        - bcmi-training: Multi-session BCI training data
        - bcmi-testing: Online BCI testing with target-measure-change
        - bcmi-tempo: Tempo control via motor imagery
        - bcmi-scores: Movie score emotion induction
        - bcmi-fmri: Joint EEG-fMRI classical music study
    """
    path = Path(dataset_path)
    dataset_name = path.name.lower()
    
    loader_map = {
        'bcmi-calibration': BCMICalibrationLoader,
        'bcmi-training': BCMITrainingLoader,
        'bcmi-testing': BCMITestingLoader,
        'bcmi-tempo': BCMITempoLoader,
        'bcmi-scores': BCMIScoresLoader,
        'bcmi-fmri': BCMIFMRILoader
    }
    
    if dataset_name in loader_map:
        return loader_map[dataset_name](dataset_path)
    else:
        # Fallback to base calibration loader
        print(f"Warning: Unknown dataset '{dataset_name}', using calibration loader")
        return BCMICalibrationLoader(dataset_path)


def load_all_bcmi_datasets(bcmi_root: str, max_subjects_per_dataset: int = 3) -> Dict[str, BaseBCMILoader]:
    """
    Load all BCMI datasets from a root directory containing multiple datasets.
    
    This convenience function automatically discovers and loads all BCMI datasets
    in a directory, providing a comprehensive view of the entire BCMI collection.
    Useful for comparative analysis across different experimental paradigms.
    
    Args:
        bcmi_root: Path to directory containing BCMI dataset subdirectories.
                  Expected structure:
                  bcmi_root/
                  ├── bcmi-calibration/
                  ├── bcmi-training/
                  ├── bcmi-testing/
                  ├── bcmi-tempo/
                  ├── bcmi-scores/
                  └── bcmi-fmri/
        max_subjects_per_dataset: Maximum number of subjects to load per dataset
                                 for memory management and quick exploration
        
    Returns:
        Dictionary mapping dataset names to their respective loaded loaders:
        {
            'bcmi-calibration': BCMICalibrationLoader,
            'bcmi-training': BCMITrainingLoader,
            ...
        }
        
    Example:
        >>> # Load all datasets with limited subjects for exploration
        >>> loaders = load_all_bcmi_datasets('/path/to/bcmi', max_subjects_per_dataset=2)
        >>> 
        >>> # Compare dataset characteristics
        >>> for name, loader in loaders.items():
        ...     print(f"\n=== {name.upper()} ===")
        ...     loader.get_dataset_statistics()
        >>> 
        >>> # Cross-dataset emotion analysis
        >>> all_happy_trials = []
        >>> for name, loader in loaders.items():
        ...     happy_trials = loader.get_condition_trials(1)  # Happy/excited
        ...     all_happy_trials.extend(happy_trials)
        >>> print(f"Total happy trials across all datasets: {len(all_happy_trials)}")
        >>> 
        >>> # Dataset-specific analysis
        >>> cal_loader = loaders['bcmi-calibration']
        >>> train_loader = loaders['bcmi-training']
        >>> 
        >>> # Compare calibration vs training data
        >>> cal_subjects = set(cal_loader.data.keys())
        >>> train_subjects = set(train_loader.data.keys())
        >>> overlap = cal_subjects.intersection(train_subjects)
        >>> print(f"Subjects in both calibration and training: {overlap}")
        
    Performance Notes:
        - Loading all datasets can be memory intensive
        - Use max_subjects_per_dataset to limit memory usage
        - Consider loading specific datasets individually for detailed analysis
    """
    bcmi_path = Path(bcmi_root)
    loaders = {}
    
    # Find all BCMI dataset directories
    dataset_dirs = [d for d in bcmi_path.iterdir() 
                   if d.is_dir() and d.name.startswith('bcmi-')]
    
    print(f"🔄 Loading all BCMI datasets from {bcmi_root}")
    print(f"📦 Found {len(dataset_dirs)} datasets")
    print("=" * 60)
    
    for dataset_dir in sorted(dataset_dirs):
        try:
            print(f"\n📂 Loading {dataset_dir.name}...")
            loader = create_bcmi_loader(str(dataset_dir))
            loader.load_all_subjects(max_subjects=max_subjects_per_dataset, verbose=False)
            loaders[dataset_dir.name] = loader
            
            # Brief statistics
            n_subjects = len(loader.data)
            total_trials = sum(
                sum(run_data.get('n_trials', 0) 
                   for session_data in subject_data.values()
                   for run_data in session_data.values())
                for subject_data in loader.data.values()
            )
            print(f"✅ {dataset_dir.name}: {n_subjects} subjects, {total_trials} trials")
            
        except Exception as e:
            print(f"❌ {dataset_dir.name}: Failed - {str(e)[:50]}...")
    
    print(f"\n🎉 Successfully loaded {len(loaders)} BCMI datasets!")
    return loaders


if __name__ == "__main__":
    # Example usage
    print("🎵 BCMI Dataset Loaders")
    print("=" * 50)
    
    # Load single dataset
    calibration_path = "/home/zmrocze/studia/uwr/magisterka/datasets/bcmi/bcmi-calibration"
    
    if Path(calibration_path).exists():
        print("📦 Loading BCMI Calibration dataset...")
        loader = create_bcmi_loader(calibration_path)
        data = loader.load_all_subjects(max_subjects=2, max_runs_per_session=2)
        loader.get_dataset_statistics()
        
        print(f"\n💡 USAGE EXAMPLES:")
        print(f"# Create loader for specific dataset:")
        print(f"# loader = create_bcmi_loader('/path/to/bcmi-calibration')")
        print(f"# ")
        print(f"# Load all subjects:")
        print(f"# data = loader.load_all_subjects()")
        print(f"# ")
        print(f"# Get trials for specific emotion:")
        print(f"# happy_trials = loader.get_condition_trials(1)")
        print(f"# ")
        print(f"# Load all BCMI datasets:")
        print(f"# loaders = load_all_bcmi_datasets('/path/to/bcmi')")
    else:
        print(f"Dataset not found at {calibration_path}")
        print("Please check the path and try again.")

