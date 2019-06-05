# Fandak: فندک

## Ideas

* Config files:
    - One idea is to have config files (python files) with a specific signature that implements a specific function.
    - These files should be able to be loaded dynamically.
    - These files should be able to be saved in snapshot directories for resuming.
    - But it is confusing for me how to manage the dependencies for these config files.
    - I think 3 config files are needed: Train Dataset, Model, Training
    - We can have one more config file for: Evaluation. 