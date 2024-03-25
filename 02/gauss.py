from common import matrixType, print_matrix


def gauss_factorization(A: matrixType, b: matrixType):
    """Perform Gauss factorization on matrix A and vertical vector b"""
    n = len(A)
    A = [[A[i][j] for j in range(n)] for i in range(n)]
    b = [b[i] for i in range(n)]

    # Forward elimination
    for i in range(n):
        for j in range(i + 1, n):
            factor = A[j][i] / A[i][i]
            for k in range(i, n):
                A[j][k] -= factor * A[i][k]
            b[j][0] -= factor * b[i][0]

    # Backward substitution
    for j in range(n - 1, 0, -1):
        for i in range(j - 1, -1, -1):
            factor = A[i][j] / A[j][j]

            for k in range(j, n):
                A[i][k] -= factor * A[j][k]
            b[i][0] -= factor * b[j][0]
        b[j][0] /= A[j][j]
        A[j][j] = 1.0
    return A, b


if __name__ == "__main__":
    print("\nGauss:")
    A = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 10.0]]
    b = [[1.0], [2.0], [3.0]]
    print("Matrix A:")
    print_matrix(A)
    print("Vector b:")
    print(b)
    A, b = gauss_factorization(A, b)
    print("Matrix A after factorization:")
    print_matrix(A)
    print("Vector b after factorization:")
    print(b)
