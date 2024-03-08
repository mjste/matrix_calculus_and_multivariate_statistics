from typing import Tuple

import numpy as np

from utils import matrixType, timeit


def multiply_matrices(A: matrixType, B: matrixType) -> Tuple[matrixType, int, int]:
    """Multiply two matrices using the standard algorithm
    Args:
    A: First matrix
    B: Second matrix

    Returns:
    C: Resultant matrix
    sum_count: Number of additions
    multiply_count: Number of multiplications
    """
    if len(A[0]) != len(B):
        raise ValueError("Matrix dimensions do not match")

    rows = len(A)
    cols = len(B[0])
    max_k = len(B)

    C = [[0 for _ in range(cols)] for _ in range(rows)]
    sum_count = 0
    multiply_count = 0

    for i in range(rows):
        for j in range(cols):
            for k in range(max_k):
                C[i][j] += A[i][k] * B[k][j]
                multiply_count += 1
                sum_count += 1

    return C, sum_count, multiply_count


def partition_square_matrix(
        A: matrixType,
) -> Tuple[matrixType, matrixType, matrixType, matrixType]:
    """Partition a matrix into 4 submatrices
    Args:
    A: Input matrix

    Returns:
    A11: Top left submatrix
    A12: Top right submatrix
    A21: Bottom left submatrix
    A22: Bottom right submatrix
    """
    k = len(A)
    A11 = [[A[i][j] for j in range(k // 2)] for i in range(k // 2)]
    A12 = [[A[i][j] for j in range(k // 2, k)] for i in range(k // 2)]
    A21 = [[A[i][j] for j in range(k // 2)] for i in range(k // 2, k)]
    A22 = [[A[i][j] for j in range(k // 2, k)] for i in range(k // 2, k)]
    return A11, A12, A21, A22


def binet_multiplication(A: matrixType, B: matrixType) -> Tuple[matrixType, int, int]:
    """Multiply two matrices using Binet's method.
    Matrice dimensions must be powers of 2 and equal.

    Args:
    A: First matrix
    B: Second matrix

    Returns:
    C: Resultant matrix
    sum_count: Number of additions
    multiply_count: Number of multiplications
    """

    if not len(A) == len(A[0]) == len(B) == len(B[0]):
        raise ValueError("Matrix dimensions do not match")
    k = len(A)
    if not is_power_of_two(k):
        raise ValueError("Matrix dimensions are not powers of 2")

    if k == 1:
        return [[A[0][0] * B[0][0]]], 0, 1

    A11, A12, A21, A22 = partition_square_matrix(A)
    B11, B12, B21, B22 = partition_square_matrix(B)

    C = [[0 for _ in range(k)] for _ in range(k)]

    A11B11, s1, m1 = binet_multiplication(A11, B11)
    A12B21, s2, m2 = binet_multiplication(A12, B21)
    A11B12, s3, m3 = binet_multiplication(A11, B12)
    A12B22, s4, m4 = binet_multiplication(A12, B22)
    A21B11, s5, m5 = binet_multiplication(A21, B11)
    A22B21, s6, m6 = binet_multiplication(A22, B21)
    A21B12, s7, m7 = binet_multiplication(A21, B12)
    A22B22, s8, m8 = binet_multiplication(A22, B22)

    for i in range(k // 2):
        for j in range(k // 2):
            C[i][j] += A11B11[i][j] + A12B21[i][j]
        for j in range(k - k // 2):
            C[i][k // 2 + j] += A11B12[i][j] + A12B22[i][j]

    for i in range(k - k // 2):
        for j in range(k // 2):
            C[k // 2 + i][j] += A21B11[i][j] + A22B21[i][j]
        for j in range(k - k // 2):
            C[k // 2 + i][k // 2 + j] += A21B12[i][j] + A22B22[i][j]

    sum_count = s1 + s2 + s3 + s4 + s5 + s6 + s7 + s8 + k * k
    multiply_count = m1 + m2 + m3 + m4 + m5 + m6 + m7 + m8

    return C, sum_count, multiply_count


def is_power_of_two(n: int) -> bool:
    """Check if a number is a power of 2
    Args:
    n: Number to check

    Returns:
    True if n is a power of 2, False otherwise
    """
    return n != 0 and (n & (n - 1)) == 0


def add_mat(A: matrixType, B: matrixType) -> matrixType:
    return [[A[i][j] + B[i][j] for j in range(len(A[0]))] for i in range(len(A))]


def sub_mat(A: matrixType, B: matrixType) -> matrixType:
    return [[A[i][j] - B[i][j] for j in range(len(A[0]))] for i in range(len(A))]


def strassen_multiplication(A: matrixType, B: matrixType) -> Tuple[matrixType, int, int]:
    """Multiply two matrices using Strassens's method.

    Args:
    A: First matrix
    B: Second matrix

    Returns:
    C: Resultant matrix
    sum_count: Number of additions
    multiply_count: Number of multiplications
    """
    k = len(A)
    if k == 1:
        return [[A[0][0] * B[0][0]]], 0, 1

    A11, A12, A21, A22 = partition_square_matrix(A)
    B11, B12, B21, B22 = partition_square_matrix(B)

    adds, muls = 0, 0
    p = len(A11) ** 2  # count of add operations in each partition add/sub

    P1, a, m = strassen_multiplication(A11, sub_mat(B12, B22))
    adds += a + p;
    muls += m
    P2, a, m = strassen_multiplication(add_mat(A11, A12), B22)
    adds += a + p;
    muls += m
    P3, a, m = strassen_multiplication(add_mat(A21, A22), B11)
    adds += a + p;
    muls += m
    P4, a, m = strassen_multiplication(A22, sub_mat(B21, B11))
    adds += a + p;
    muls += m
    P5, a, m = strassen_multiplication(add_mat(A11, A22), add_mat(B11, B22))
    adds += a + 2 * p;
    muls += m
    P6, a, m = strassen_multiplication(sub_mat(A12, A22), add_mat(B21, B22))
    adds += a + 2 * p;
    muls += m
    P7, a, m = strassen_multiplication(sub_mat(A11, A21), add_mat(B11, B12))
    adds += a + 2 * p;
    muls += m

    C11 = add_mat(sub_mat(add_mat(P5, P4), P2), P6)
    adds += 3 * p
    C12 = add_mat(P1, P2)
    adds += p
    C21 = add_mat(P3, P4)
    adds += p
    C22 = sub_mat(sub_mat(add_mat(P5, P1), P3), P7)
    adds += 3 * p

    C = [[0 for _ in range(k)] for _ in range(k)]
    for i in range(k // 2):
        for j in range(k // 2):
            C[i][j] = C11[i][j]
            C[i][k // 2 + j] = C12[i][j]
            C[k // 2 + i][j] = C21[i][j]
            C[k // 2 + i][k // 2 + j] = C22[i][j]

    return C, adds, muls


if __name__ == "__main__":
    timed_multiply_matrices = timeit(multiply_matrices)
    timed_binet_multiplication = timeit(binet_multiplication)

    for exponent in range(1, 8):
        size = 2 ** exponent
        np_array_1 = np.random.randint(1, 100, size=(size, size))
        np_array_2 = np.random.randint(1, 100, size=(size, size))
        array_1 = np_array_1.tolist()
        array_2 = np_array_2.tolist()

        np_true_result = np_array_1 @ np_array_2

        print(f"Matrix Multiplication using standard method for {size}x{size} matrix:")
        C, sum_count, multiply_count = timed_multiply_matrices(array_1, array_2)
        assert np.allclose(np.array(C), np_true_result)
        print(f"Sum Count: {sum_count}")
        print(f"Multiply Count: {multiply_count}")
        print()

        print(f"Matrix Multiplication using Binet's Method for {size}x{size} matrix:")
        C, sum_count, multiply_count = timed_binet_multiplication(array_1, array_2)
        assert np.allclose(np.array(C), np_true_result)
        print(f"Sum Count: {sum_count}")
        print(f"Multiply Count: {multiply_count}")
        print()

        print(f"Matrix Multiplication using Strassen's Method for {size}x{size} matrix:")
        C, sum_count, multiply_count = timeit(strassen_multiplication)(array_1, array_2)
        assert np.allclose(np.array(C), np_true_result)
        print(f"Sum Count: {sum_count}")
        print(f"Multiply Count: {multiply_count}")
        print()
