# BCMI Dataset Processing - Technical Guidelines

## Quick Reference

### Dataset Identification
```
bcmi-training:    Emotion pairs (20s+20s), 360 WAV files
bcmi-calibration: Single emotions (21s), 216 WAV files  
bcmi-testing:     Real-time BCI, no music files
bcmi-scores:      Movie scores (20s), 720 MP3 files
bcmi-tempo:       Motor control, no music files
bcmi-fmri:        Mixed stimuli, 233 files, code 47 = music onset
```

### Event Code Mapping (1-9)
```
1: LVLA (Low/Low)      4: LVNA (Low/Neutral)    7: LVHA (Low/High)
2: NVLA (Neutral/Low)  5: NVNA (Neutral/Neutral) 8: NVHA (Neutral/High)*
3: HVLA (High/Low)     6: HVNA (High/Neutral)    9: HVHA (High/High)

*Code 8 missing in calibration/testing
```

## Implementation Checklist

### 1. Dataset Detection
```python
def detect_dataset(path):
    if "training" in path: return "training"
    if "calibration" in path: return "calibration"
    # etc...
```

### 2. Event Loading
```python
# All datasets use TSV format with: onset, duration, trial_type
events = pd.read_csv(event_file, sep='\t')
```

### 3. Music File Mapping

#### Training Dataset
- Files: `{emotion1}-{emotion2}_{variant}.wav`
- Logic: Pair consecutive events (20s apart)
- Example: Events 1,6 → File `1-6_1.wav`

#### Calibration Dataset  
- Files: `{valence}{arousal}a{number}.wav`
- Logic: Map emotion code to VA string
- Example: Code 3 → `hvla` → File `hvla1.wav`

#### fMRI Dataset
- Code 47 = music onset
- Three subdirs: `generated/`, `classical/`, `washout/`
- Task determines which subdir to use

#### Scores Dataset
- Files in `likely_stimuli/Set1/`
- Use `set1_tracklist.csv` for mapping
- Single event per trial (code 0)

### 4. Timing Parameters
```python
DURATIONS = {
    'training': 21,
    'calibration': 21,
    'testing': 21,
    'scores': 20,
    'tempo': 20,
    'fmri': 0.001  # Instantaneous markers
}

BUFFER_PRE = 2   # seconds before onset
BUFFER_POST = 2  # seconds after offset
SAMPLING_RATE = 1000  # Hz (all datasets)
```

### 5. EEG Cutting
```python
def cut_eeg(event, dataset_type):
    start = event['onset'] - BUFFER_PRE
    end = event['onset'] + DURATIONS[dataset_type] + BUFFER_POST
    return start, end
```

### 6. Special Cases

#### Simultaneous Events (Calibration)
- Emotion code (1-9) and control code (>100) have same onset
- Use emotion code for music mapping, ignore control

#### Empty Events (Tempo)
- Most event files are empty
- Focus on motor imagery analysis, not music

#### Dense Events (fMRI)
- Thousands of events per file
- Filter by trial_type: 47 (music), 265 (rating), 10 (n-back)

## Processing Pipeline

```python
def process_bcmi_dataset(dataset_path):
    # 1. Identify dataset type
    dataset_type = detect_dataset(dataset_path)
    
    # 2. Load events
    event_files = find_event_files(dataset_path)
    events = load_events(event_files)
    
    # 3. Find music files
    music_files = find_music_files(dataset_path)
    
    # 4. For each event:
    for event in events:
        # Get music file
        music_file = map_event_to_music(event, dataset_type)
        
        # Cut EEG segment
        eeg_start, eeg_end = cut_eeg(event, dataset_type)
        
        # Align music (starts at event onset)
        music_start = 0
        music_offset = event['onset']
        
        # Process aligned data
        process_aligned_data(eeg_segment, music_segment)
```

## Common Pitfalls

1. **Don't assume all datasets have music files** - Testing/Tempo don't
2. **Check for simultaneous events** - Calibration has paired markers
3. **Handle empty event files** - Common in Tempo dataset
4. **Use correct duration** - 21s vs 20s vs 0.001s
5. **Map fMRI code 47** - It's music onset, not emotion
6. **Scores uses code 0** - Not emotion codes 1-9
7. **Training needs pairs** - Single event insufficient for music selection

## File Structure
```
datasets/bcmi/
├── bcmi-training/
│   ├── sub-*/ses-*/eeg/*_events.tsv
│   └── stimuli/*.wav (360 files)
├── bcmi-calibration/
│   ├── sub-*/eeg/*_events.tsv  
│   └── stimuli/*.wav (216 files)
├── bcmi-testing/
│   └── sub-*/eeg/*_events.tsv (no stimuli)
├── bcmi-scores/
│   ├── sub-*/eeg/*_events.tsv
│   └── likely_stimuli/Set1/*.mp3 (720 files)
├── bcmi-tempo/
│   └── sub-*/eeg/*_events.tsv (no stimuli)
└── bcmi-fmri/
    ├── sub-*/eeg/*_events.tsv
    └── stimuli/
        ├── generated/*.wav (216)
        ├── classical/*.mp3 (7)
        └── washout/*.mp3 (10)
```

## Testing Your Implementation

```python
# Verify counts
assert count_music_files('training') == 360
assert count_music_files('calibration') == 216
assert count_music_files('testing') == 0

# Check event durations
assert all(event['duration'] == 21 for event in training_events)
assert all(event['duration'] == 0.001 for event in fmri_events)

# Validate emotion codes
valid_codes = set(range(1, 10))
assert all(e['trial_type'] in valid_codes for e in emotion_events)
```

This guide provides the essential technical information for implementing BCMI dataset processing. Refer to `bcmi_datasets_explainer.md` for detailed explanations and edge cases.
