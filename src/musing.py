import mne
import mne_bids
from mne_bids import BIDSPath, read_raw_bids, get_entity_vals
import pandas as pd
import numpy as np
from pathlib import Path
import warnings

# Enhanced song information with research context
songs_info_enhanced = {
    1: {"name": "Trip to the lonely planet", "artist": "Mark Alow", "genre": "Deep House", 
        "duration": 125, "tempo": 121.95, "characteristics": "Electronic, Western"},
    2: {"name": "Sail", "artist": "Awolnation", "genre": "Indie", 
        "duration": 114, "tempo": 119, "characteristics": "Rock, English lyrics"},
    3: {"name": "Concept 15", "artist": "Kodomo", "genre": "Electronics", 
        "duration": 132, "tempo": 161, "characteristics": "Experimental electronic"},
    4: {"name": "Aurore", "artist": "Claire David", "genre": "New Age", 
        "duration": 111, "tempo": None, "characteristics": "Ambient, meditative"},
    5: {"name": "Proof", "artist": "Idiotape", "genre": "Electronic Dance", 
        "duration": 124, "tempo": 123, "characteristics": "Danceable electronic"},
    6: {"name": "Glider", "artist": "Tycho", "genre": "Ambient", 
        "duration": 100, "tempo": 126, "characteristics": "Atmospheric, instrumental"},
    7: {"name": "Raag Bihag", "artist": "B.Sivaramakrishna Rao", "genre": "Hindustani Classical", 
        "duration": 116, "tempo": 70, "characteristics": "Indian classical, traditional"},
    8: {"name": "Albela sajan", "artist": "Ismail Darbar", "genre": "Indian Semi-Classical", 
        "duration": 121, "tempo": 194, "characteristics": "Indian, Hindi lyrics"},
    9: {"name": "Mor Bani Thanghat Kare", "artist": "Sanjay Leela Bhansali", "genre": "Indian Folk", 
        "duration": 126, "tempo": 117, "characteristics": "Indian folk, Gujarati lyrics"},
    10: {"name": "Fallin", "artist": "Dr. SaxLove", "genre": "Soft Jazz", 
         "duration": 129, "tempo": 197, "characteristics": "Jazz, instrumental"},
    11: {"name": "Master of Running", "artist": "Rickeyabo", "genre": "Goth Rock", 
         "duration": 113, "tempo": 120, "characteristics": "Dark, alternative rock"},
    12: {"name": "JB", "artist": "Nobody.one", "genre": "Progressive Instrumental Rock", 
         "duration": 117, "tempo": 146, "characteristics": "Complex, instrumental"}
}


# Create comprehensive dataset class
class MUSINGDataset:
    def __init__(self, bids_root):
        self.bids_root = bids_root
        self.subjects = get_entity_vals(bids_root, 'subject')
        self.sessions = get_entity_vals(bids_root, 'session')
        self.songs_info = songs_info_enhanced
        behavioral_file = Path(bids_root) / 'stimuli' / 'Behavioural_data'
        self.behavioral_data = pd.read_csv(behavioral_file, sep='\t')
        self.all_data = {}
        
    def load_complete_dataset(self, max_subjects=None, verbose=True):
        """Load EEG data and behavioral ratings for all subjects and sessions"""
        subjects_to_load = self.subjects[:max_subjects] if max_subjects else self.subjects
        
        print(f"🔄 Loading complete MUSIN-G dataset...")
        print(f"📦 Subjects to load: {len(subjects_to_load)} of {len(self.subjects)}")
        print()
        
        for i, subject in enumerate(subjects_to_load, 1):
            if verbose:
                print(f"Subject {subject} ({i}/{len(subjects_to_load)}):")
            
            self.all_data[subject] = {}
            
            for session in self.sessions:
                # Load EEG data - Fix run number format (remove zero padding)
                bids_path = BIDSPath(
                    subject=subject,
                    session=session,
                    task='MusicListening',
                    run=int(session),  # Use integer instead of zero-padded string
                    datatype='eeg',
                    root=self.bids_root
                )
                
                try:
                    raw = read_raw_bids(bids_path, verbose=False)
                    
                    # Get song info
                    song_info = self.songs_info[int(session)]
                    
                    # Get behavioral ratings for this subject-song combination
                    ratings = self.behavioral_data[
                        (self.behavioral_data['Subject'] == int(subject)) & 
                        (self.behavioral_data['Song_ID'] == int(session))
                    ]
                    
                    if not ratings.empty:
                        enjoyment = ratings['Enjoyment'].iloc[0]
                        familiarity = ratings['Familiarity'].iloc[0]
                    else:
                        enjoyment, familiarity = None, None
                    
                    # Enhanced metadata - store in info['temp'] to avoid MNE restrictions
                    raw.info['description'] = f"Song {session}: {song_info['name']} - {song_info['artist']}"
                    
                    # Use temp storage for custom metadata (MNE-compatible way)
                    if 'temp' not in raw.info:
                        raw.info['temp'] = {}
                    
                    raw.info['temp']['song_metadata'] = {
                        'genre': song_info['genre'],
                        'tempo': song_info['tempo'],
                        'characteristics': song_info['characteristics'],
                        'enjoyment_rating': enjoyment,
                        'familiarity_rating': familiarity,
                        'song_name': song_info['name'],
                        'artist': song_info['artist']
                    }
                    
                    # Store complete information
                    self.all_data[subject][session] = {
                        'raw': raw,
                        'song_info': song_info,
                        'enjoyment': enjoyment,
                        'familiarity': familiarity,
                        'file_path': str(bids_path.fpath)
                    }
                    
                    if verbose:
                        print(f"  ✓ Song {session}: {song_info['name']} ({song_info['genre']}) | "
                              f"Enjoy: {enjoyment}/5, Familiar: {familiarity}/5")
                        
                except Exception as e:
                    if verbose:
                        error_msg = str(e)
                        if "File does not exist" in error_msg:
                            # Extract suggestions from error message
                            lines = error_msg.split('\n')
                            if len(lines) > 2 and "Did you mean" in lines[1]:
                                print(f"  ✗ Song {session}: File naming issue - trying run number without zero padding")
                            else:
                                print(f"  ✗ Song {session}: File not found")
                        else:
                            print(f"  ✗ Song {session}: {error_msg[:100]}...")
            
            if verbose:
                print()
        
        return self.all_data
    
    def get_dataset_statistics(self):
        """Get comprehensive dataset statistics"""
        if not self.all_data:
            print("No data loaded. Call load_complete_dataset() first.")
            return
        
        # Count successful loads
        total_files = 0
        successful_loads = 0
        
        for subject in self.all_data:
            for session in self.all_data[subject]:
                total_files += 1
                if 'raw' in self.all_data[subject][session]:
                    successful_loads += 1
        
        print(f"📈 DATASET STATISTICS:")
        print(f"• Total expected files: {len(self.subjects)} × {len(self.sessions)} = {len(self.subjects) * len(self.sessions)}")
        print(f"• Successfully loaded: {successful_loads}/{total_files}")
        print(f"• Success rate: {successful_loads/total_files*100:.1f}%")
        
        # Behavioral statistics by genre
        print(f"\n🎭 GENRE-WISE BEHAVIORAL STATISTICS:")
        for session_num, song_info in self.songs_info.items():
            genre_ratings = self.behavioral_data[self.behavioral_data['Song_ID'] == session_num]
            if not genre_ratings.empty:
                mean_enjoy = genre_ratings['Enjoyment'].mean()
                mean_familiar = genre_ratings['Familiarity'].mean()
                print(f"  {song_info['genre']:25} | Enjoy: {mean_enjoy:.1f}/5 | Familiar: {mean_familiar:.1f}/5")
    
    def get_subject_data(self, subject_id):
        """Get all data for a specific subject"""
        return self.all_data.get(subject_id, {})
    
    def get_song_data_across_subjects(self, session_id):
        """Get data for a specific song across all subjects"""
        song_data = {}
        for subject in self.all_data:
            if session_id in self.all_data[subject]:
                song_data[subject] = self.all_data[subject][session_id]
        return song_data
    
    def get_song_metadata(self, raw_object):
        """Helper function to extract song metadata from MNE Raw object"""
        if 'temp' in raw_object.info and 'song_metadata' in raw_object.info['temp']:
            return raw_object.info['temp']['song_metadata']
        return None
