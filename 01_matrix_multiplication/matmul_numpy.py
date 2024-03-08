import numpy as np


def normal_matmul(A, B):
    assert A.shape[1] == B.shape[0], "Incompatible matrix dimensions for multiplication."

    C = np.zeros((A.shape[0], B.shape[1]))

    for i in range(A.shape[0]):
        for j in range(B.shape[1]):
            for k in range(A.shape[1]):
                C[i, j] += A[i, k] * B[k, j]

    return C


def split_matrix(A):
    row, col = A.shape
    row2, col2 = row // 2, col // 2
    return A[:row2, :col2], A[:row2, col2:], A[row2:, :col2], A[row2:, col2:]


def strassen_matmul(A, B):
    if len(A) == 1:
        return A * B

    a11, a12, a21, a22 = split_matrix(A)
    b11, b12, b21, b22 = split_matrix(B)

    p1 = strassen_matmul(a11, b12 - b22)
    p2 = strassen_matmul(a11 + a12, b22)
    p3 = strassen_matmul(a21 + a22, b11)
    p4 = strassen_matmul(a22, b21 - b11)
    p5 = strassen_matmul(a11 + a22, b11 + b22)
    p6 = strassen_matmul(a12 - a22, b21 + b22)
    p7 = strassen_matmul(a11 - a21, b11 + b12)

    c11 = p5 + p4 - p2 + p6
    c12 = p1 + p2
    c21 = p3 + p4
    c22 = p5 + p1 - p3 - p7

    C = np.vstack((np.hstack((c11, c12)), np.hstack((c21, c22))))
    return C


def is_power_of_two(n):
    return np.log2(n).is_integer()


def binet_matmul(A, B):
    # Ensure the matrix dimensions are correct
    assert A.shape == B.shape, "Matrices dimensions do not match"
    assert A.shape[0] == A.shape[1], "Input matrices must be square"
    assert is_power_of_two(A.shape[0]), "Dimensions must be power of 2"

    n = A.shape[0]

    if n == 1:
        return A * B

    A11, A12, A21, A22 = split_matrix(A)
    B11, B12, B21, B22 = split_matrix(B)

    P1 = binet_matmul(A11, B11)
    P2 = binet_matmul(A12, B21)
    P3 = binet_matmul(A11, B12)
    P4 = binet_matmul(A12, B22)
    P5 = binet_matmul(A21, B11)
    P6 = binet_matmul(A22, B21)
    P7 = binet_matmul(A21, B12)
    P8 = binet_matmul(A22, B22)

    C11 = P1 + P2
    C12 = P3 + P4
    C21 = P5 + P6
    C22 = P7 + P8

    C = np.vstack((np.hstack((C11, C12)), np.hstack((C21, C22))))
    return C


def binet_normal_mixed_matmul(A, B, l):
    """Multiply two matrices using a mixed method of Binet's and normal multiplication.

    If matrix dims <= (2**l, 2**l), normal multiplication is used.
    Otherwise, Binet's method is used.
    """
    if A.shape[0] <= 2 ** l and A.shape[1] <= 2 ** l and B.shape[1] <= 2 ** l:
        return normal_matmul(A, B)
    else:
        return binet_matmul(A, B)


def strassen_normal_mixed_matmul(A, B, l):
    """Multiply two matrices using a mixed method of Strassen's and normal multiplication.

    If matrix dims <= (2**l, 2**l), normal multiplication is used.
    Otherwise, Strassen's method is used.
    """
    if A.shape[0] <= 2 ** l and A.shape[1] <= 2 ** l and B.shape[1] <= 2 ** l:
        return normal_matmul(A, B)
    else:
        return strassen_matmul(A, B)


def binet_strassen_mixed_matmul(A, B, l):
    """Multiply two matrices using a mixed method of Binet's and Strassen's multiplication.

    If matrix dims <= (2**l, 2**l), Binet's multiplication is used.
    Otherwise, Strassen's method is used.
    """
    if A.shape[0] <= 2 ** l and A.shape[1] <= 2 ** l and B.shape[1] <= 2 ** l:
        return binet_matmul(A, B)
    else:
        return strassen_matmul(A, B)


def strassen_binet_mixed_matmul(A, B, l):
    """Multiply two matrices using a mixed method of Strassen's and Binet's multiplication.

    If matrix dims <= (2**l, 2**l), Strassen's multiplication is used.
    Otherwise, Binet's method is used.
    """
    if A.shape[0] <= 2 ** l and A.shape[1] <= 2 ** l and B.shape[1] <= 2 ** l:
        return strassen_matmul(A, B)
    else:
        return binet_matmul(A, B)
