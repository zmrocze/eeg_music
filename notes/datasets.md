
# Datasets

## description

perplexity link: https://www.perplexity.ai/search/link-me-all-available-datasets-Q5.63_7hS1WkWPHXT62n6g

1. [x] NMED-T

39GB, 10 songs, 20 participants, songs as links (stereo, mono?)

download link: https://exhibits.stanford.edu/data/catalog/jn859kj8079

paper: https://archives.ismir.net/ismir2017/paper/000198.pdf

preprocessed as: First, data from each electrode in the electrodes-by-time data matrix were zero-phase filtered using 8th-order Butterworth highpass (0.3 Hz) and notch (59–61 Hz) filters, and a 16th-order Chebyshev Type I lowpass (50 Hz) filter. Following this, the filtered data were temporally downsampled by a factor of 8 to a final sampling rate of 125 Hz.
After preprocessing all recordings, we aggregated the data on a per-song basis. The data frame for each song is thus a 3D electrodes-by-time-by-participant matrix of size 125 × T × 20

2. NMED-H

15GB, 48 adults, hindi vocal pop songs, 4 songs x 4 versions (original, measure-shuffled, phase-scrambled, reversed), 
Twelve participants were assigned to each stimulus, and each participant heard their assigned stimuli twice (24 trials total per stimulus).

download link: https://exhibits.stanford.edu/data/catalog/sd922db3535

3. torcheeg.DEAPDataset (music known?)

128Hz, 40 1-minute long excerpts from music video

4. torcheeg.DREAMERDataset (music known?)

128Hz, 18 movie clips

5. openMIIR

6. MUSIN-G

10GB

## TASKS

1. [ ] sampling rates