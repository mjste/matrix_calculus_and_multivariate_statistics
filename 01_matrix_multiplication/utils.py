import time
from typing import List, Callable

matrixType = List[List[int]]


def print_arr(A: matrixType) -> None:
    for row in A:
        print(row)


def timeit(func: Callable) -> Callable:
    """Decorator to time a function"""
    def timed(*args, **kw):
        start = time.time()
        result = func(*args, **kw)
        end = time.time()
        print(f"{func.__name__} took {end - start:.4f} seconds")
        return result

    return timed
