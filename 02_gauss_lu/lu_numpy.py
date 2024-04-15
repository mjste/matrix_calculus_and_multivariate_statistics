
import numpy as np

def LU_inplace(A):
    for i in range(A.shape[1]):
        f = A[i + 1:, i] / A[i, i]
        A[i + 1:, i:] -= A[i, i:] * f.reshape(-1, 1)
        A[i + 1:, i] = f
    return np.tril(A, -1) + np.eye(A.shape[0]), np.triu(A)


def LU(A):
    n = A.shape[0]
    U = A.copy()
    L = np.eye(n)

    for i in range(n):
        for j in range(i+1, n):
            L[j, i] = U[j, i] / U[i, i]
            U[j, i:] = U[j, i:] - L[j, i]*U[i, i:]

    return L, U

def LU_partial_pivoting(A):
    n = A.shape[0]
    U = A.copy()
    L = np.eye(n)
    P = np.eye(n)

    for i in range(n):
        pivot_row = np.argmax(abs(U[i:, i])) + i
        U[[i, pivot_row], :] = U[[pivot_row, i], :]
        L[[i, pivot_row], :i] = L[[pivot_row, i], :i]
        P[[i, pivot_row], :] = P[[pivot_row, i], :]

        for j in range(i+1, n):
            L[j, i] = U[j, i] / U[i, i]
            U[j, i:] = U[j, i:] - L[j, i]*U[i, i:]

    return P, L, U


if __name__ == "__main__":
    A = np.random.rand(5, 5)
    print("Matrix A:")
    print(A)
    L, U = LU(A.copy())
    print("Matrix L:")
    print(L)
    print("Matrix U:")
    print(U)
    print("Matrix L * U:")
    print(np.matmul(L, U))
    print("Matrix A - L * U:")
    print(A - np.matmul(L, U))

    print("\nLU with Partial Pivoting:")
    print("Matrix A:")
    print(A)
    P, L, U = LU_partial_pivoting(A.copy())
    print("Matrix P:")
    print(P)
    print("Matrix L:")
    print(L)
    print("Matrix U:")
    print(U)
    print("Matrix P * A - L * U:")
    print(np.matmul(P, A) - np.matmul(L, U))