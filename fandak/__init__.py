from .core.datasets import Dataset
from .core.evaluators import Evaluator
from .core.models import Model
from .core.trainers import Trainer

name = "fandak"

__version__ = "0.0.12"

__all__ = ["__version__", "Dataset", "Model", "Trainer", "Evaluator"]
