# Release Notes

## (April 14, 2020) version 0.1.2.1
* Added plot function to `LRVisualizer`.

## (April 10, 2020) version 0.1.2
* Fixed bug in `Evaluator`. Now the `Trainer` calls a function names `reset_storage` after each time it evaluates.
By default `reset_storage` calls `set_storage`.
*  Added `LRVisualizer`. This is a "fake" PyTorch optimizer that you can use with most of PyTorch `lr_scheduler`s.
Then you can easily see the value of the learning rate for each epoch.
