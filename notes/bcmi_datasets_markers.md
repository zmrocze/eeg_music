# BCMI Datasets Event Markers Comprehensive Guide

## Overview

The Brain-Computer Music Interface (BCMI) datasets from OpenNeuro contain 6 different datasets designed for affective music brain-computer interface research. Each dataset uses a systematic marker system to identify different emotional states and experimental conditions.

## Common Marker System Across Datasets

### Core Emotion Codes (1-9): 3×3 Valence-Arousal Grid
```
               Low Arousal    Neutral Arousal    High Arousal
High Valence:      2              3                1
                (Peaceful)    (Pleasant)        (Happy)
                
Neutral Val:       8              9                7
                (Calm)        (Neutral)         (Alert)
                
Low Valence:       5              6                4
                (Sad)       (Unpleasant)      (Angry)
```

### Control/System Markers (100+)
- **Purpose**: Technical markers for experimental control
- **Range**: Codes ≥ 100 (e.g., 104, 109, 134, 150, 163, etc.)
- **Function**: Start/stop signals, calibration markers, system events

## Dataset-Specific Analysis

### 1. BCMI-Training (ds002724)
**Purpose**: Training sessions for BCI emotion discrimination learning

**Experimental Design**:
- **Trial Structure**: Dual-target trials (20s + 20s = 40s total)
- **Training Paradigm**: Consecutive emotion pairs for contrast learning
- **Sessions**: Multiple longitudinal training sessions per subject

**Event Markers**:
- **Emotion Codes**: 1-9 (all emotions represented)
- **Control Markers**: None (pure emotion training)
- **Timing**: ~21s duration per emotion event
- **Pattern**: Emotion pairs with ~20s intervals within pairs, ~37-38s gaps between pairs

**Music Alignment**:
- **Audio Files**: 216 synthetic music files in `stimuli/` directory
- **Naming Convention**: `{emotion1}-{emotion2}_{variant}.wav` (e.g., `1-6_1.wav`)
- **Duration**: Each file ~40s (matching 20s+20s structure)
- **File Count**: 72 unique emotion pairs × 3 variants each = 216 files
- **Variants**: Exactly 3 variants (1, 2, 3) per emotion pair

**Cutting Guidelines**:
1. **Trial Start**: Use emotion event onset time (e.g., 8.865s)
2. **Trial End**: Add 40s to start time (covers both emotions + transition)
3. **Music Selection**: Match event codes to audio filename pattern

### 2. BCMI-Calibration (ds002722)
**Purpose**: System calibration for individual BCI adaptation

**Experimental Design**:
- **Trial Structure**: 40s music clips targeting specific emotional states
- **Target**: First 20s target state 1, remaining 20s target state 2
- **Sessions**: 1 session with 5 runs of 18 trials each

**Event Markers**:
- **Emotion Codes**: 1, 2, 4, 5, 6, 9 (subset of full grid)
- **Control Markers**: Extensive use (104, 109, 134, 150, 153, 163, 165, 173, 176, 182, 185, 186, 193, 194, 201, 202, 205, 206)
- **Timing**: 21s duration events
- **Pattern**: Simultaneous emotion + control markers (same onset time)

**Music Alignment**:
- **Audio Files**: 108 synthetic music files organized by valence-arousal combinations
- **Naming Convention**: `{valence}{arousal}a{number}.wav`
  - `hv` = High Valence, `lv` = Low Valence, `nv` = Neutral Valence
  - `ha` = High Arousal, `la` = Low Arousal, `na` = Neutral Arousal
- **Examples**: `hvha1.wav` (High Valence, High Arousal, variant 1)
- **File Count**: 9 VA combinations × 12 variants each = 108 files
- **Variants**: Exactly 12 variants (1-12) per VA combination
- **Duration**: ~21s files (matching event duration)

**Cutting Guidelines**:
1. **Trial Start**: Use emotion event onset
2. **Trial Duration**: 21s (single emotion state)
3. **Music Selection**: Map emotion code to valence-arousal filename
4. **Control Markers**: Ignore for music alignment (technical markers only)

### 3. BCMI-Testing (ds002723)
**Purpose**: Real-time BCI testing with feedback

**Experimental Design**:
- **Trial Structure**: 20s+20s+20s (target → measure → change)
- **Feedback**: Real-time music generation based on detected emotion
- **Control**: Participants attempt to modulate their emotional state
- **BCI Loop**: Continuous emotion detection → music parameter adjustment
- **Learning**: Participants learn to control music through emotional state changes

**Event Markers**:
- **Emotion Codes**: 2, 3, 4, 5, 6, 7, 9 (focused subset, excluding corners 1 and 8)
- **Control Markers**: None observed
- **Timing**: 21s duration events
- **Pattern**: Single emotion events with varying inter-trial intervals
- **Coverage**: Emphasizes mid-range emotions for easier BCI control

**Music Alignment**:
- **Audio Files**: None (real-time generation only)
- **Generation**: Synthetic music created on-the-fly using BCI feedback
- **Control**: Music parameters (tempo, harmony, rhythm) respond to detected emotional state
- **Adaptation**: Music changes dynamically based on participant's neural signals

**Cutting Guidelines**:
1. **Trial Start**: Use emotion event onset
2. **Trial Duration**: 21s per emotion state
3. **Music**: No pre-recorded stimuli (generated real-time)
4. **Analysis Focus**: Neural adaptation to feedback rather than stimulus response

### 4. BCMI-Scores (ds002721)
**Purpose**: Neural correlates of music-induced emotions using movie scores

**Experimental Design**:
- **Stimuli**: Pre-existing movie score excerpts (not synthetic)
- **Focus**: Natural emotional responses to real orchestral music
- **Duration**: Short clips designed to evoke specific emotions
- **Paradigm**: Passive listening to emotionally evocative film music

**Event Markers**:
- **Emotion Codes**: Very limited (only code 0 observed)
- **Control Markers**: None
- **Timing**: 20s duration events
- **Pattern**: Minimal event structure (likely single stimulus onset per recording)

**Music Alignment**:
- **Audio Files**: Movie score excerpts (no stimuli directory found)
- **Duration**: ~20s clips
- **Type**: Real orchestral music from film scores
- **Source**: Likely embedded in EEG recordings or external stimulus presentation

**Cutting Guidelines**:
1. **Trial Start**: Use event onset (3s observed)
2. **Trial Duration**: 20s
3. **Music**: External movie score excerpts (not included in dataset)
4. **Note**: Limited event structure suggests simple stimulus presentation paradigm

### 5. BCMI-Tempo (ds002720)
**Purpose**: Motor imagery BCI for tempo control

**Experimental Design**:
- **Control Paradigm**: Kinesthetic imagery (squeeze ball = increase, relax = decrease tempo)
- **Focus**: Motor control rather than emotion
- **Motor Imagery**: Right hand ball squeezing vs. relaxation
- **Calibration**: 30 trials in pairs (increase/decrease tempo)
- **Runs**: 9 total runs (1 calibration + 8 control runs)

**Event Markers**:
- **Emotion Codes**: None (empty events file)
- **Control Markers**: None
- **Motor Commands**: Likely encoded differently (not in standard event files)
- **Pattern**: No events recorded in BIDS structure

**Music Alignment**:
- **Audio Files**: None available in dataset
- **Generation**: Real-time tempo modulation of synthetic music
- **Control**: User controls music tempo via motor imagery BCI
- **Feedback**: Music tempo increases/decreases based on detected motor intentions

**Cutting Guidelines**:
- **Not applicable**: No event markers available
- **Alternative**: Use motor imagery paradigm timing (if available in raw data annotations)
- **Focus**: Motor-related neural activity rather than music-evoked responses

### 6. BCMI-fMRI (ds002725)
**Purpose**: Joint EEG-fMRI during affective music listening

**Experimental Design**:
- **Modality**: Simultaneous EEG and fMRI recording
- **Task Sequence**: 5 separate tasks per participant
  1. `genMusic01` (synthetic music block 1)
  2. `genMusic02` (synthetic music block 2) 
  3. `genMusic03` (synthetic music block 3)
  4. `classicalMusic` (classical music pieces)
  5. `washout` (animal sounds for baseline)
- **Paradigm**: Music listening with continuous emotion rating (FEELTRACE)
- **Cognitive Task**: Embedded n-back working memory task during classical music
- **FEELTRACE**: Real-time 2D valence-arousal emotion reporting

**Event Markers - Precise Mapping**:
- **Code 1**: Task/session start marker (appears once per task)
- **Code 47**: Music stimulus onset (primary alignment marker)
  - GenMusic01: 307 events, GenMusic02: 298 events, GenMusic03: ~300 events
  - ClassicalMusic: 949 events, Washout: 53 events
- **Code 265**: Participant emotion ratings via FEELTRACE
  - GenMusic01: 305 events, GenMusic02: 346 events
  - ClassicalMusic: 587 events, Washout: 49 events
- **Code 10**: N-back cognitive task events (during classical music)
  - GenMusic01: 273 events, GenMusic02: 275 events
  - ClassicalMusic: 930 events, Washout: 34 events
- **Code 768**: Classical music specific markers (12-24 per task)
- **Codes 1283, 1285, 34053**: Task boundary/session control markers

**Precise Experimental Sequence**:
1. **Generated Music Blocks (genMusic01-03)**:
   - **Music Files**: Select from 216 files (`{emotion1}-{emotion2}_{variant}.wav`)
   - **Event Code 47**: Marks onset of each ~40s music clip
   - **Event Code 265**: Continuous FEELTRACE emotion ratings
   - **Event Code 10**: Embedded n-back task prompts
   - **File Selection**: Emotion codes likely embedded in stimulus presentation order

2. **Classical Music Block**:
   - **Music Files**: 7 classical pieces (120-180s each)
   - **Event Code 47**: Music segment onsets (949 events = multiple segments per piece)
   - **Event Code 768**: Classical-specific timing markers
   - **Event Code 10**: Intensive n-back task (930 events)
   - **Event Code 265**: Continuous emotion ratings throughout

3. **Washout Block**:
   - **Music Files**: 10 animal sounds (1-2s each)
   - **Event Code 47**: Animal sound onsets (53 events)
   - **Event Code 265**: Minimal ratings (49 events)
   - **Purpose**: Baseline/transition between emotional stimuli

**Music-to-Event Alignment**:
- **Code 47 = Music Onset**: Primary synchronization marker for all stimulus types
- **Generated Music**: Code 47 onset → select corresponding `{emotion1}-{emotion2}_{variant}.wav`
- **Classical Music**: Code 47 onset → segment within one of 7 classical pieces
- **Washout**: Code 47 onset → specific animal sound from 10 available files
- **Multiple Code 47 per File**: Long classical pieces have multiple onsets (segmented presentation)

**Music Alignment**:
- **Audio Files**: Three distinct categories (233 total files)
  - `generated/`: 216 synthetic music files using emotion pair naming (`1-2_1.wav`, etc.)
  - `classical/`: 7 classical music pieces (Chopin, Rachmaninoff, Mendelssohn, Beethoven)
  - `washout/`: 10 animal sound files for inter-block transitions
- **Naming Conventions**:
  - Generated: `{emotion1}-{emotion2}_{variant}.wav` (same as BCMI-Training)
  - Classical: `p{number}_{composer-piece-performer}.mp3`
  - Washout: `{animal}.mp3` (cat, dog, cow, etc.)
- **Duration**: 
  - Generated: ~40s (7.1MB WAV files)
  - Classical: ~120-180s (2.0-2.6MB MP3 files)
  - Washout: ~1-2s (10-18KB MP3 files)

**Cutting Guidelines - Precise Protocol**:
1. **Generated Music Trials**:
   - **Onset Marker**: Use Code 47 events
   - **Duration**: 40s per trial (similar to BCMI-Training)
   - **Music Selection**: Match to `{emotion1}-{emotion2}_{variant}.wav` files
   - **Emotion Ratings**: Code 265 events provide continuous valence-arousal ratings
   
2. **Classical Music Trials**:
   - **Onset Marker**: Use Code 47 events (multiple per piece)
   - **Duration**: Variable segments within 120-180s pieces
   - **Music Selection**: Match to one of 7 classical MP3 files
   - **Segmentation**: Code 47 indicates segment boundaries within long pieces
   - **N-back Task**: Code 10 events indicate cognitive task periods
   
3. **Washout Periods**:
   - **Onset Marker**: Use Code 47 events
   - **Duration**: 1-2s animal sounds
   - **Purpose**: Baseline/reset between emotional stimuli
   - **File Selection**: Match to specific animal sound from 10 MP3 files

4. **fMRI-Specific Considerations**:
   - **Hemodynamic Response**: Add 6s delay for BOLD signal
   - **Block Design**: Use Code 47 for stimulus blocks
   - **Continuous Ratings**: Code 265 for FEELTRACE emotion tracking
   - **Task Boundaries**: Codes 1283, 1285, 34053 for session structure

5. **Multi-Task Analysis**:
   - **Task Order**: genMusic01 → genMusic02 → genMusic03 → classicalMusic → washout
   - **Cross-Task**: Compare emotion responses across synthetic vs. classical music
   - **Cognitive Load**: N-back task effects on music processing (Code 10 events)

## General Cutting and Alignment Guidelines

### For EEG Analysis:
1. **Pre-stimulus Buffer**: Add 2-3s before event onset
2. **Post-stimulus Buffer**: Add 2-3s after event end
3. **Baseline**: Use pre-stimulus period for baseline correction
4. **Artifacts**: Check for movement artifacts during transitions

### For Music-EEG Synchronization:
1. **Onset Precision**: Event onset marks exact music start time
2. **Duration Matching**: Event duration = music clip duration
3. **Sampling Rate**: Ensure music is resampled to match EEG sampling (1000Hz)
4. **Alignment Check**: Verify music and EEG start simultaneously

### Dataset-Specific Recommendations:

**Best for Music-EEG Analysis**:
- **BCMI-Training**: Complete music stimuli with clear emotion pairs
- **BCMI-Calibration**: Systematic valence-arousal coverage
- **BCMI-Scores**: Real music (not synthetic)

**Limited Music Data**:
- **BCMI-Testing**: Real-time generation only
- **BCMI-Tempo**: No music files available
- **BCMI-fMRI**: Mixed stimuli types

### Code Example for Event Cutting:
```python
# For BCMI-Training
emotion_onset = 8.865  # seconds
trial_duration = 40    # seconds (20s + 20s)
eeg_start = emotion_onset - 2  # 2s pre-stimulus
eeg_end = emotion_onset + trial_duration + 2  # 2s post-stimulus

# For BCMI-Calibration  
emotion_onset = 9.643  # seconds
trial_duration = 21    # seconds
eeg_start = emotion_onset - 2
eeg_end = emotion_onset + trial_duration + 2

# For BCMI-fMRI - Generated Music
music_onset = 0.021    # Code 47 event onset
trial_duration = 40    # seconds
eeg_start = music_onset - 2
eeg_end = music_onset + trial_duration + 2
# BOLD signal (add 6s delay for fMRI analysis)
fmri_start = music_onset + 6
fmri_end = music_onset + trial_duration + 6

# For BCMI-fMRI - Classical Music Segments
classical_onset = 12.337  # Code 47 event onset
# Duration varies - use next Code 47 or end of piece
next_onset = 15.321    # Next Code 47 event
segment_duration = next_onset - classical_onset
eeg_start = classical_onset - 2
eeg_end = classical_onset + segment_duration + 2

# For BCMI-fMRI - Emotion Ratings Alignment
rating_events = [12.613, 13.64, 15.694]  # Code 265 events
for rating_time in rating_events:
    # Extract FEELTRACE emotion values at these timepoints
    valence_arousal = extract_emotion_rating(rating_time)
```

## Summary

The BCMI datasets provide a comprehensive framework for studying music-emotion-brain interactions with systematic emotional state targeting via the 3×3 valence-arousal grid. The key insight is that emotion codes (1-9) represent target emotional states, while control markers (100+) are technical annotations. For music-EEG analysis, use event onset times to align with corresponding audio stimuli, ensuring proper temporal synchronization between neural responses and musical features.

## Comprehensive Dataset Comparison Table

| Dataset | Purpose | Emotion Codes | Control Markers | Audio Files | Trial Duration | Music Type | Best Use Case |
|---------|---------|---------------|-----------------|-------------|----------------|------------|---------------|
| **BCMI-Training** | BCI emotion discrimination training | 1-9 (all emotions) | None | 216 synthetic files | 40s (20s+20s pairs) | Synthetic (emotion pairs) | **Optimal for music-EEG analysis** |
| **BCMI-Calibration** | Individual BCI system calibration | 1,2,4,5,6,9 (subset) | Extensive (100+) | 108 synthetic files | 21s (single emotion) | Synthetic (valence-arousal) | **Good for systematic emotion coverage** |
| **BCMI-Testing** | Real-time BCI feedback testing | 2,3,4,5,6,7,9 (mid-range) | None | None (real-time) | 21s | Real-time generated | BCI feedback analysis |
| **BCMI-Scores** | Natural music emotion responses | 0 only (minimal) | None | None available | 20s | Movie scores (real) | **Best for natural music responses** |
| **BCMI-Tempo** | Motor imagery tempo control | None (empty) | None | None available | Variable | Real-time tempo control | Motor BCI research |
| **BCMI-fMRI** | Joint EEG-fMRI emotion study | 47 (music onset), 265 (ratings), 10 (n-back) | 1,768,1283,1285,34053 | 233 mixed files | 40s/120-180s/1-2s | Mixed (synthetic+classical+animal) | **Excellent for multimodal analysis** |

### Audio File Summary by Dataset

| Dataset | Total Files | File Types | Naming Convention | Duration Range | File Sizes |
|---------|-------------|------------|-------------------|----------------|------------|
| **BCMI-Training** | 216 | WAV | `{emotion1}-{emotion2}_{variant}.wav` (variants 1-3) | ~40s | 7.1MB |
| **BCMI-Calibration** | 108 | WAV | `{valence}{arousal}a{number}.wav` (variants 1-12) | ~21s | 3.5-5.4MB |
| **BCMI-Testing** | 0 | - | Real-time generation | - | - |
| **BCMI-Scores** | 0 | - | External movie scores | ~20s | - |
| **BCMI-Tempo** | 0 | - | Real-time tempo control | Variable | - |
| **BCMI-fMRI** | 233 | WAV/MP3 | Mixed conventions | 1s-180s | 10KB-7.7MB |
| └─ Generated | 216 | WAV | `{emotion1}-{emotion2}_{variant}.wav` (variants 1-3) | ~40s | 7.1MB |
| └─ Classical | 7 | MP3 | `p{number}_{composer-piece}.mp3` | 120-180s | 2.0-2.6MB |
| └─ Washout | 10 | MP3 | `{animal}.mp3` | 1-2s | 10-18KB |

### Experimental Design Comparison

| Dataset | Trial Structure | Paradigm | Sessions | Participants | EEG Setup |
|---------|-----------------|----------|----------|--------------|-----------|
| **BCMI-Training** | 20s+20s pairs | Emotion contrast learning | Multiple (longitudinal) | Variable | 1000Hz, 37ch |
| **BCMI-Calibration** | 40s single emotion | Individual calibration | 1 session, 5 runs, 18 trials | 19 | 1000Hz, 37ch |
| **BCMI-Testing** | 20s+20s+20s | Real-time feedback control | Variable | Variable | 1000Hz, 37ch |
| **BCMI-Scores** | Single presentation | Passive listening | 1 session | Variable | 1000Hz, 37ch |
| **BCMI-Tempo** | Motor imagery blocks | Tempo control via BCI | 9 runs (1 calib + 8 control) | 19 | 1000Hz, 37ch |
| **BCMI-fMRI** | Block design | Passive + n-back tasks | 3 runs (10 min each) | 21 | 1000Hz + fMRI |

### Recommendations by Research Goal

| Research Goal | Recommended Dataset(s) | Reason |
|---------------|------------------------|---------|
| **Music-EEG Feature Analysis** | BCMI-Training, BCMI-fMRI | Complete audio stimuli with systematic emotion targeting |
| **Emotion Classification** | BCMI-Calibration, BCMI-Training | Systematic valence-arousal coverage with clear labels |
| **Real Music Studies** | BCMI-Scores, BCMI-fMRI (classical) | Non-synthetic, naturalistic music stimuli |
| **BCI Development** | BCMI-Testing, BCMI-Calibration | Real-time feedback and calibration data |
| **Multimodal Analysis** | BCMI-fMRI | Joint EEG-fMRI with comprehensive stimuli |
| **Longitudinal Learning** | BCMI-Training | Multiple sessions tracking BCI learning progression |
