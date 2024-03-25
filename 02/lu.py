import numpy as np
from typing import Tuple

from common import matrixType, print_matrix


def lu_factorization(A: matrixType) -> Tuple[matrixType, matrixType]:
    n = len(A)
    L = [[0.0] * n for _ in range(n)]
    U = [[0.0] * n for _ in range(n)]

    for j in range(n):
        L[j][j] = 1.0
        for i in range(j + 1):
            sum_u = sum(U[k][j] * L[i][k] for k in range(i))
            U[i][j] = A[i][j] - sum_u
        for i in range(j + 1, n):
            sum_l = sum(L[i][k] * U[k][j] for k in range(j))
            L[i][j] = (A[i][j] - sum_l) / U[j][j]
    return L, U


def lu_factorization_with_pivoting(A: matrixType) -> Tuple[matrixType, matrixType, matrixType]:
    n = len(A)
    L = [[0.0] * n for _ in range(n)]
    U = [[0.0] * n for _ in range(n)]
    P = [[float(i == j) for i in range(n)] for j in range(n)]  # Initialize the permutation matrix as an identity matrix

    for j in range(n):
        # Partial pivoting
        pivot_value = abs(A[j][j])
        pivot_row = j
        for i in range(j+1, n):
            if abs(A[i][j]) > pivot_value:
                pivot_value = abs(A[i][j])
                pivot_row = i
        if pivot_row != j:
            # Swap rows in A, P
            A[j], A[pivot_row] = A[pivot_row], A[j]
            P[j], P[pivot_row] = P[pivot_row], P[j]

        # Proceed with LU decomposition
        L[j][j] = 1.0
        for i in range(j + 1):
            sum_u = sum(U[k][j] * L[i][k] for k in range(i))
            U[i][j] = A[i][j] - sum_u
        for i in range(j + 1, n):
            sum_l = sum(L[i][k] * U[k][j] for k in range(j))
            L[i][j] = (A[i][j] - sum_l) / U[j][j]

    return L, U, P


if __name__ == "__main__":
    print("\nLU:")
    A = np.random.rand(3, 3).astype(float)
    print("Matrix A:")
    print_matrix(A)
    L, U = lu_factorization(A)
    print("Matrix L:")
    print_matrix(L)
    print("Matrix U:")
    print_matrix(U)
    print("Matrix L * U:")
    print_matrix(np.matmul(L, U))
    print("Matrix A - L * U:")
    print_matrix(np.subtract(A, np.matmul(L, U)))
    print("--------------------")
    print("\nLU with PIVOTING:")
    print("Matrix A:")
    print_matrix(A)
    L, U, P = lu_factorization_with_pivoting(A)
    print("Matrix L:")
    print_matrix(L)
    print("Matrix U:")
    print_matrix(U)
    print("Matrix P:")
    print_matrix(P)
    print("Matrix L * U:")
    print_matrix(np.matmul(L, U))
    print("Matrix P * A - L * U:")
    print_matrix(np.subtract(np.matmul(P, A), np.matmul(L, U)))
