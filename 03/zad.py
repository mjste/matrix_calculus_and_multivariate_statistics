import numpy as np

M = np.array([[4, 9, 2], [3, 5, 7], [8, 1, 6]])


def SVD(A: np.ndarray):
    left_shape = A.shape[0]
    right_shape = A.shape[1]
    eigenvalues, V = np.linalg.eigh(np.dot(A.T, A))

    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    V = V[:, idx]

    singular_values = np.sqrt(eigenvalues)
    right_singular_vectors = V

    left_singular_vectors = np.dot(A, right_singular_vectors)

    with np.errstate(divide="ignore"):
        left_singular_vectors /= singular_values

    left_singular_vectors = left_singular_vectors[:, :left_shape]
    Sigma = np.diag(singular_values)[:left_shape, :right_shape]

    return left_singular_vectors, Sigma, right_singular_vectors.T


def matrix_norm_1(A: np.ndarray):
    return np.max(np.sum(np.abs(A), axis=0))


def matrix_norm_inf(A: np.ndarray):
    return np.max(np.sum(np.abs(A), axis=1))


def matrix_norm_2(A: np.ndarray):
    return np.max(np.abs(np.linalg.eigvals(A)))


A = np.array([[1, 2, 0], [2, 0, 2]])
B = A.T

U, S, Vt = SVD(A)

print("U")
print(U)
print("S")
print(S)
print("Vt")
print(Vt)

print("A")
print(U @ S @ Vt)


for n in range(2, 5):
    for m in range(2, 5):
        for k in range(1000):
            A = np.random.rand(n, m)
            U, S, Vt = SVD(A)
            result = np.allclose(A, U @ S @ Vt)
            if not result:
                print("FAILED")
                print(A)
                print(U @ S @ Vt)


print("Matrix norm 1")
print(matrix_norm_1(M))
print(np.linalg.norm(M, ord=1))
print()

print("Matrix norm inf")
print(matrix_norm_inf(M))
print(np.linalg.norm(M, ord=np.inf))
print()

print("Matrix norm 2")
print(matrix_norm_2(M))
print(np.linalg.norm(M, ord=2))
print()


print("Matrix norm 1")
print(matrix_norm_1(A))
print(np.linalg.norm(A, ord=1))
print()

print("Matrix norm inf")
print(matrix_norm_inf(A))
print(np.linalg.norm(A, ord=np.inf))
print()

print("Matrix norm 2")
print(matrix_norm_2(A))
print(np.linalg.norm(A, ord=2))
print()
