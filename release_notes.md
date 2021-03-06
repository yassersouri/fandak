# Release Notes

## (August 4, 2020) version 0.1.3
* Add lock file to `_figure_run_number` in `Trainer` so it becomes thread safe. Used `oslu_concurrency` for this functionality.

## (July 10, 2020) version 0.1.2.8
* Add `set_value` functionality for `ScalarMetricCollection`. This could be used for setting some value after training and saving it.

## (July 10, 2020) version 0.1.2.7
* Changed the behaviour of `_save_info_of_run` in the `Trainer`.
    * It now saves better markdown info.
    * It now assumes that `cfg` has `.dump` function. (This was also assumed before, now it is more explicit)

## (July 3, 2020) version 0.1.2.6
* Add default `__repr__` implementation for `GeneralDataClass`.

## (June 26, 2020) version 0.1.2.5
* Change GeneralBatch's `default_collate` so that non-`Tensor` types are also collated correctly.

## (June 25, 2020) version 0.1.2.4
* Removing explicit dependency for pytorch vision greater than 1.1

## (April 14, 2020) version 0.1.2.3
* (**bug fix**) Moved around creation of the SummaryWriter in the `Trainer` so that `load_training` will not create a new entry
in Tensorboard. This also meant that I had to change the location of `create_metrics` call.

## (April 14, 2020) version 0.1.2.1
* Added plot function to `LRVisualizer`.

## (April 10, 2020) version 0.1.2
* Fixed bug in `Evaluator`. Now the `Trainer` calls a function names `reset_storage` after each time it evaluates.
By default `reset_storage` calls `set_storage`.
*  Added `LRVisualizer`. This is a "fake" PyTorch optimizer that you can use with most of PyTorch `lr_scheduler`s.
Then you can easily see the value of the learning rate for each epoch.
