import warnings
from concurrent.futures import ProcessPoolExecutor
from typing import Iterator

import numpy as np
from tqdm import tqdm


def smart_map(func: callable, items: list, single_threaded: bool, desc: str = ''):
    if single_threaded:
        # Do a simple loop
        for graph in tqdm(items, desc=desc):
            yield func(graph)
        return

    with ProcessPoolExecutor() as executor:
        # When we multiprocess, objects are pickled and copied in the child process
        # instead of using the same object, so we have to return objects from the
        # function to get the changes back
        futures = [
            executor.submit(func, *item) if isinstance(item, tuple)
            else executor.submit(func, item)
            for item in items
        ]
        with tqdm(total=len(futures), desc=desc) as pbar:
            for future in futures:
                yield future.result()
                pbar.update(1)


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


def get_cycle(path: list[str]) -> str:
    last = path[-1]
    first = None
    for i, t in enumerate(path):
        if t == last:
            first = i
            break
    assert first is not None, "Cycle doesn't exist, but get_cycle was called"

    return " ".join(path[first:])
