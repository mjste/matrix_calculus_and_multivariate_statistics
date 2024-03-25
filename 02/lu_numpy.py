
import numpy as np

def LU(A):
    for i in range(A.shape[1]):
        f = A[i + 1:, i] / A[i, i]
        A[i + 1:, i:] -= A[i, i:] * f.reshape(-1, 1)
        A[i + 1:, i] = f
    return np.tril(A, -1) + np.eye(A.shape[0]), np.triu(A)

if __name__ == "__main__":
    A = np.random.randint(1, 10, (3, 3)).astype(float)
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