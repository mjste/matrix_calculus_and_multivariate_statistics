import numpy as np
from typing import List, Tuple

matrixType = List[List[float]]

def print_matrix(A: matrixType) -> None:
    for row in A:
        print("[ " + " ".join(f'{x:.2f}' for x in row) + " ]")


def lu_factorization(A: matrixType) -> Tuple[matrixType, matrixType]:
    n = len(A)
    L = [[0.0] * n for _ in range(n)]
    U = [[0.0] * n for _ in range(n)]

    for j in range(n):
        L[j][j] = 1.0
        for i in range(j + 1):
            sum_u = sum(U[k][j] * L[i][k] for k in range(i))
            U[i][j] = A[i][j] - sum_u
        for i in range(j, n):
            sum_l = sum(U[j][k] * L[i][k] for k in range(j))
            L[i][j] = (A[i][j] - sum_l) / U[j][j]
    return L, U


def lu_factorization_with_pivoting(A: matrixType) -> Tuple[matrixType, matrixType, matrixType]:
    n = len(A)
    L = [[0.0] * n for _ in range(n)]
    U = [[0.0] * n for _ in range(n)]
    P = [[float(i == j) for i in range(n)] for j in range(n)]

    for j in range(n):
        row = max(range(j, n), key=lambda i: abs(A[i][j]))
        if j != row:
            A[j], A[row] = A[row], A[j]
            P[j], P[row] = P[row], P[j]

        L[j][j] = 1.0
        for i in range(j + 1):
            sum_u = sum(U[k][j] * L[i][k] for k in range(i))
            U[i][j] = A[i][j] - sum_u
        for i in range(j + 1, n):
            sum_l = sum(U[j][k] * L[i][k] for k in range(j))
            L[i][j] = (A[i][j] - sum_l) / U[j][j]
    return L, U, P


if __name__ == "__main__":
    A = np.random.randint(1, 10, (3, 3))
    L, U = lu_factorization(A)
    print("Matrix A:")
    print_matrix(A)
    print("Matrix L:")
    print_matrix(L)
    print("Matrix U:")
    print_matrix(U)
    print("Matrix L * U:")
    print_matrix(np.matmul(L, U))
    print("Matrix A - L * U:")
    print_matrix(np.subtract(A, np.matmul(L, U)))
