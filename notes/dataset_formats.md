
## interface to dataset loader

Every dataset loader should allow below operations.

Methods for providing dataset summary:

1. load_all_subjects()

 - operation that loads the full dataset into memory. like implemention in @src/bcmi.py

2. get_dataset_statistics()

 - like implementation in @src/bcmi.py

Methods allowing to load eeg fragments to memory one by one:

3. get_dataset_structure()

 - returns a dictionary of form:
  { <user_id> : } 