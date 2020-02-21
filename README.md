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
python -m fandak.hyper /path/to/root metric1 [metric2] [--exp-name baseline-*]
```

If no exp-name is provided, then all are going to be considered.
exp-name can be something usable from glob.