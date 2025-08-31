# Complete working OpenMIIR loader based on the simple version
from mne_bids import get_entity_vals
from mne_bids import BIDSPath, read_raw_bids
import pandas as pd
        
class OpenMIIRLoader:
    def __init__(self, root_path):
        self.root_path = root_path
        self.subjects = get_entity_vals(root_path, 'subject')
        
        # Get available runs from tasks (run1, run2, etc.)
        try:
            available_tasks = get_entity_vals(root_path, 'task')
            self.runs = [task[3:] for task in available_tasks if task.startswith('run')]
            self.runs = sorted(self.runs) if self.runs else ['1', '2', '3', '4', '5']
        except:
            self.runs = ['1', '2', '3', '4', '5']
            
        self.data = {}
        
        # Emotional state mapping (3x3 valence x arousal)
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
    
    def load_subject_data(self, subject_id, max_runs=None):
        """Load data for one subject"""
        # Use all runs if max_runs not specified
        if max_runs is None:
            max_runs = len(self.runs)
        
        self.data[subject_id] = {}
        print(f"Loading subject {subject_id}:")
        
        for run in self.runs[:max_runs]:
            try:
                # Load EEG data
                bids_path = BIDSPath(
                    subject=subject_id,
                    task=f'run{run}',
                    datatype='eeg',
                    root=self.root_path
                )
                
                raw = read_raw_bids(bids_path, verbose=False)
                
                # Load events
                events_path = bids_path.copy().update(suffix='events', extension='.tsv')
                events_df = pd.read_csv(events_path.fpath, sep='\t')
                
                # Enhanced event processing
                processed_events = self._process_events(events_df)
                
                # Store data
                self.data[subject_id][run] = {
                    'raw': raw,
                    'events': events_df,
                    'processed_events': processed_events,
                    'duration': raw.times[-1],
                    'n_channels': raw.info['nchan'],
                    'sfreq': raw.info['sfreq'],
                    'n_trials': len(processed_events['condition_events']) if processed_events else 0
                }
                
                trial_count = len(processed_events['condition_events']) if processed_events else 0
                print(f"  ✓ Run {run}: {raw.times[-1]:.1f}s, {len(events_df)} events, {trial_count} trials")
                
            except Exception as e:
                print(f"  ✗ Run {run}: {str(e)[:50]}...")
                
        return self.data[subject_id]
    
    def _process_events(self, events_df):
        """Process events to separate conditions from markers"""
        try:
            # Separate condition events (1-9) from timing markers (100+)
            condition_events = events_df[events_df['trial_type'] <= 9].copy()
            marker_events = events_df[events_df['trial_type'] > 100].copy()
            
            # Add emotional state information to condition events
            condition_events['emotion_description'] = condition_events['trial_type'].map(
                lambda x: self.emotional_states.get(x, {}).get('description', 'Unknown')
            )
            condition_events['valence'] = condition_events['trial_type'].map(
                lambda x: self.emotional_states.get(x, {}).get('valence', 'Unknown')
            )
            condition_events['arousal'] = condition_events['trial_type'].map(
                lambda x: self.emotional_states.get(x, {}).get('arousal', 'Unknown')
            )
            
            return {
                'condition_events': condition_events,
                'marker_events': marker_events,
                'all_events': events_df
            }
        except Exception as e:
            print(f"    Warning: Could not process events - {str(e)[:50]}")
            return None
    
    def load_all_subjects(self, max_subjects=None, max_runs_per_subject=None, verbose=True):
        """Load data for all available subjects"""
        subjects_to_load = self.subjects[:max_subjects] if max_subjects else self.subjects
        
        if verbose:
            print(f"🔄 Loading OpenMIIR dataset...")
            print(f"📦 Subjects to load: {len(subjects_to_load)} of {len(self.subjects)}")
            print(f"🏃 Runs per subject: {max_runs_per_subject or len(self.runs)}")
            print()
        
        successful_subjects = 0
        failed_subjects = 0
        
        for subject in subjects_to_load:
            try:
                self.load_subject_data(subject, max_runs_per_subject)
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
            print(f"• Success rate: {successful_subjects/(successful_subjects+failed_subjects)*100:.1f}%")
        
        return self.data
    
    def get_dataset_statistics(self):
        """Get comprehensive dataset statistics"""
        if not self.data:
            print("No data loaded. Call load_all_subjects() or load_subject_data() first.")
            return
        
        total_runs = 0
        total_trials = 0
        all_conditions = []
        
        print(f"📈 DATASET STATISTICS:")
        print(f"• Loaded subjects: {len(self.data)}")
        
        for subject_id, subject_data in self.data.items():
            subject_runs = len(subject_data)
            subject_trials = sum(run_data.get('n_trials', 0) for run_data in subject_data.values())
            total_runs += subject_runs
            total_trials += subject_trials
            
            # Collect condition information
            for run_data in subject_data.values():
                if run_data.get('processed_events') and 'condition_events' in run_data['processed_events']:
                    conditions = run_data['processed_events']['condition_events']['trial_type'].tolist()
                    all_conditions.extend(conditions)
        
        print(f"• Total runs loaded: {total_runs}")
        print(f"• Total trials: {total_trials}")
        
        if all_conditions:
            import pandas as pd
            condition_counts = pd.Series(all_conditions).value_counts().sort_index()
            print(f"\n🎭 EMOTIONAL CONDITION DISTRIBUTION:")
            for condition, count in condition_counts.items():
                print(f"  Condition {condition}: {count:3d} trials")
        
        # Sample technical info
        if self.data:
            sample_subject = list(self.data.keys())[0]
            sample_run = list(self.data[sample_subject].keys())[0]
            sample_data = self.data[sample_subject][sample_run]
            
            print(f"\n⚙️  TECHNICAL SPECIFICATIONS:")
            print(f"• Sampling rate: {sample_data['sfreq']} Hz")
            print(f"• Channels per recording: {sample_data['n_channels']}")
            print(f"• Run duration: ~{sample_data['duration']:.0f} seconds")
    
    def get_subject_data(self, subject_id):
        """Get all data for a specific subject"""
        return self.data.get(subject_id, {})
    
    def get_condition_trials(self, condition_code):
        """Get all trials for a specific emotional condition across all subjects"""
        condition_trials = []
        
        for subject_id, subject_data in self.data.items():
            for run_id, run_data in subject_data.items():
                if (run_data.get('processed_events') and 
                    'condition_events' in run_data['processed_events']):
                    
                    condition_events = run_data['processed_events']['condition_events']
                    matching_trials = condition_events[condition_events['trial_type'] == condition_code]
                    
                    for _, trial in matching_trials.iterrows():
                        condition_trials.append({
                            'subject': subject_id,
                            'run': run_id,
                            'trial_info': trial,
                            'raw_data': run_data['raw']
                        })
        
        return condition_trials
    
    def get_available_subjects(self):
        """Get list of all available subjects"""
        return self.subjects
    
    def get_emotional_state_info(self, condition_code):
        """Get information about an emotional state"""
        return self.emotional_states.get(condition_code, {})
    
def main(root_path = '/home/zmrocze/studia/uwr/magisterka/datasets/openmiir'):
    # Initialize the OpenMIIR loader
    loader = OpenMIIRLoader('/home/zmrocze/studia/uwr/magisterka/datasets/openmiir')

    print("🎵 OpenMIIR Dataset Loader")
    print("=" * 50)
    print(f"Available subjects: {loader.get_available_subjects()[:5]}... (showing first 5)")
    print()
    
    # Load a few subjects for demonstration
    print("📦 Loading first 3 subjects (2 runs each)...")
    data = loader.load_all_subjects(max_subjects=3, max_runs_per_subject=2)
    
    # Show statistics
    loader.get_dataset_statistics()
    
    print(f"\n💡 USAGE EXAMPLES:")
    print(f"# Load all subjects:")
    print(f"# loader.load_all_subjects()")
    print(f"# ")
    print(f"# Load specific subject:")
    print(f"# subject_data = loader.load_subject_data('01')")
    print(f"# ")
    print(f"# Get trials for specific emotion:")
    print(f"# happy_trials = loader.get_condition_trials(1)  # High valence, high arousal")
    print(f"# ")
    print(f"# Access raw EEG data:")
    print(f"# raw = loader.data['01']['1']['raw']  # Subject 01, Run 1")
    print(f"# events = loader.data['01']['1']['processed_events']['condition_events']")


# Example usage
if __name__ == "__main__":
    main()