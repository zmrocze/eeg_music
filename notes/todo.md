

 - [x] start with fmri, how to read markers
 - [ ] store path information in the loaded bids (only if needed). map the trial_iterator with saving into a dir. load an ondisk dataset.
 - [ ] go into model training, figure out which and provide that dataset interface (simply indexable right?). augmentations and common preprocessing

## miscs

 - [ ] fix hardcoded dataset paths in tests
 - [ ] fix the tests after data.py interface changes. the tests use not the functionality from data as designed but write and test their own helpers. leave in background with smart warp.