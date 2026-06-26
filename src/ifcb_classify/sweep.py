"""Hyperparameter sweep expansion."""

from collections import namedtuple
from itertools import product


def generate_sweep_runs(params: dict) -> list:
    """Expand ``{param: [values]}`` into the full Cartesian grid of runs.

    Returns a list of namedtuples (one per combination), with fields named after
    the sweep parameters. For example ``{"lr": [0.1, 0.01], "epochs": [10]}``
    yields two ``Run(lr=..., epochs=10)`` tuples.
    """
    Run = namedtuple("Run", params.keys())
    return [Run(*v) for v in product(*params.values())]
