# TODOS

 - [x] Change repr system to config + system call + hash
    - [x] Dataset
    - [x] Model
    - [x] Evaluator
    - [x] Trainer
 - [x] Add yacs
 - [x] Specify the return types as dataclasses.
    - [x] Dataset
    - [x] Model
    - [x] Evaluator
    - [x] Trainer
 - [x] Specify model, dataset, trainer, etc. creation from the config file.
    - [x] Dataset
    - [x] Model
    - [x] Evaluator
    - [x] Trainer
 - [x] Add repr of experiment as `writer.add_text`
 - [x] Add overfit sampler
 - [x] Add metrics for evaluator and trainer
 - [x] Trainer fixes
    - [x] Add create optimizer and create scheduler
    - [x] Add more abstract method to implement from the config instead of adding arguments to Trainer constructor.
    - [x] Use Path instead of str.
    - [x] Automatically create a metric for main loss and each evaluation.


## Tests

 - [ ] Metrics
    - [ ] Use Mock to mock the writer and test Metrics and MetricsCollection.
 - [ ] Trainer
    - [ ] Make sure than num_epochs, etc. are correct after resumes, etc.