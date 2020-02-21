[![PyPI version](https://badge.fury.io/py/fandak.svg)](https://pypi.org/project/fandak/)
[![Travis Build](https://api.travis-ci.com/yassersouri/fandak.svg?branch=master)](https://travis-ci.com/yassersouri/fandak)

# Fandak: فندک

Coming soon...

Will help you train your models for research.

## Installation

:exclamation: Requires Python 3.7 :snake:.

`pip install fandak`

## Examples

See `examples` directory.


## Visualizing the effects of hyper-parameters

Sample usage:

```bash
python -m fandak.hyper /path/to/root metric1 [metric2] [--exp-name baseline-*] [--params-list path/to/params/list.txt]
```

If no exp-name is provided, then all are going to be considered.
exp-name can be something usable from glob.

We also have to specify the params list. What to have in the figure? Otherwise, we look
at the experiments and find the params that are not the same among them.
If a param is missing from a config file, then it is replaces with "?".