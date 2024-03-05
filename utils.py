from typing import List
import time

matrixType = List[List[int]]


def print_arr(A: matrixType) -> None:
    for row in A:
        print(row)


def timeit(func):
    def timed(*args, **kw):
        start = time.time()
        result = func(*args, **kw)
        end = time.time()
        print(f"{func.__name__} took {end - start:.4f} seconds")
        return result

    return timed
