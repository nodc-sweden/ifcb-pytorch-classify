from collections import namedtuple
from itertools import product


def generate_sweep_runs(params: dict) -> list:
    Run = namedtuple("Run", params.keys())
    return [Run(*v) for v in product(*params.values())]
