import warnings
from typing import Iterator

import numpy as np


def print_stats(name: str, samples: list) -> None:
    if len(samples) == 0:
        print(f'  {name} (N=0)')
        return
    print(f'  {name} (N={len(samples)}) - mean: {np.mean(samples)}, std: {np.std(samples)}, '
          f'min: {np.min(samples)}, max: {np.max(samples)}')


def logistic_function(x: float) -> float:
    with warnings.catch_warnings():
        warnings.filterwarnings('error', category=RuntimeWarning)
        try:
            return 1 / (1 + np.exp(x))
        except RuntimeWarning:
            return 0


def batch_list(input_list: list, batch_size: int) -> Iterator[list]:
    num_elements = len(input_list)
    for start in range(0, num_elements, batch_size):
        yield input_list[start:start + batch_size]


def json_value(value: any, type_str: str) -> dict:
    try:
        if type_str == 'string':
            value = str(value)
        elif type_str == 'long' or type_str == 'integer':
            value = int(value)
        elif type_str == 'boolean':
            value = str(value).lower() == 'true'
    except ValueError:
        pass
    except TypeError:
        pass
    return {
        'type': type_str,
        'value': value
    }
