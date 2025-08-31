
# Datasets

## description

perplexity link: https://www.perplexity.ai/search/link-me-all-available-datasets-Q5.63_7hS1WkWPHXT62n6g

1. [x] NMED-T

39GB, 10 songs, 20 participants, songs as links (stereo, mono?)

download link: https://exhibits.stanford.edu/data/catalog/jn859kj8079

paper: https://archives.ismir.net/ismir2017/paper/000198.pdf

preprocessed as: First, data from each electrode in the electrodes-by-time data matrix were zero-phase filtered using 8th-order Butterworth highpass (0.3 Hz) and notch (59–61 Hz) filters, and a 16th-order Chebyshev Type I lowpass (50 Hz) filter. Following this, the filtered data were temporally downsampled by a factor of 8 to a final sampling rate of 125 Hz.
After preprocessing all recordings, we aggregated the data on a per-song basis. The data frame for each song is thus a 3D electrodes-by-time-by-participant matrix of size 125 × T × 20

2. [x] NMED-H

15GB, 48 adults, hindi vocal pop songs, 4 songs x 4 versions (original, measure-shuffled, phase-scrambled, reversed), 
Twelve participants were assigned to each stimulus, and each participant heard their assigned stimuli twice (24 trials total per stimulus).

download link: https://exhibits.stanford.edu/data/catalog/sd922db3535

citation: Blair Kaneshiro, Duc T. Nguyen, Jacek P. Dmochowski, Anthony M. Norcia, and Jonathan Berger (2016). Naturalistic Music EEG Dataset - Hindi (NMED-H). Stanford Digital Repository. Available at: http://purl.stanford.edu/sd922db3535

3. [x] torcheeg.DEAPDataset (music known?)

128Hz, 40 1-minute long excerpts from music video

4. torcheeg.DREAMERDataset (music known?)

128Hz, 18 movie clips

5. [x] openMIIR

6. Affective Music

1. Title: EEG data investigating neural correlates of music-induced emotion.
 - https://openneuro.org/datasets/ds002721
 - movie scores
 - name: bcmi-scores
2. A dataset recorded during development of an affective brain-computer music interface: calibration session
 - https://openneuro.org/datasets/ds002722
 - calibration, synthetic music
 - 21s trials
 - name: bcmi-calibration
3. A dataset recorded during development of an affective brain-computer music interface: testing session
 - https://openneuro.org/datasets/ds002723
 - testing, synthetic music (only code provided) generated real-time
 - 20s+20s+20s - targetted affection, then measure affection, then try to change it
 - name: bcmi-testing
4. A dataset recorded during development of an affective brain-computer music interface: training sessions
 - https://openneuro.org/datasets/ds002724
 - training, synthetic music 
 - 20s+20s - trials targeting two different affections
 - name: bcmi-training
5. A dataset recorded during development of a tempo-based brain-computer music interface
 - https://openneuro.org/datasets/ds002720/
 - synthetic music (only code provided)
 - participant controls tempo with imagined movement
 - name: bcmi-tempo
6. A dataset recording joint EEG-fMRI during affective music listening
 - https://openneuro.org/datasets/ds002725
 - with fmri, synthetic music
 - three 10-min runs, participants listened to randomly-selected pieces of generated music (40s), followed by a 2 minute n-back (working mem) task, then 30-min run listening to classical music pieces (120-180s).
 - During each music-listening trial participants reported their current felt emotions via the FEELTRACE interface
 - name: bcmi-fmri

7. [x] MUSIN-G

10GB

## TASKS

1. [ ] check sampling rates, event markers, understand the experimental conditions, 
2. [ ] connect with music
3. [ ] ensure and implement common eeg processing
4. [ ] group into a single dataloader. online storage.
