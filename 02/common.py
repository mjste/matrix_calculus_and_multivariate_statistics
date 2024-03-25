from typing import List

matrixType = List[List[float]]


def print_matrix(A: matrixType) -> None:
    for row in A:
        print("[ " + " ".join(f"{x:.2f}" for x in row) + " ]")
