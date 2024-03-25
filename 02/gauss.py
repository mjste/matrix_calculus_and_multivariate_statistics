from common import matrixType, print_matrix
import numpy as np

debug = {"pivot": [], "nopivot": []}


def copy_matrix(A: matrixType) -> matrixType:
    rows = len(A)
    cols = len(A[0])
    return [[A[i][j] for j in range(cols)] for i in range(rows)]


def debug_fn(A: matrixType, b: matrixType, pivot=False):
    if pivot:
        debug["pivot"].append((copy_matrix(A), copy_matrix(b)))
    else:
        debug["nopivot"].append((copy_matrix(A), copy_matrix(b)))


def forward_pass(A: matrixType, b: matrixType, pivot=False, eps=1.0e-10):
    n = len(A)

    debug_fn(A, b, pivot)

    for i in range(n):
        if pivot:
            for j in range(i + 1, n):
                if abs(A[j][i]) > abs(A[i][i]):
                    A[i], A[j] = A[j], A[i]
                    b[i], b[j] = b[j], b[i]
        # check if A[i][i] is zero, if yes, swap with the next non-zero row
        if abs(A[i][i]) < eps:
            for j in range(i + 1, n):
                if abs(A[j][i]) > eps:
                    A[i], A[j] = A[j], A[i]
                    b[i], b[j] = b[j], b[i]
                    break
        debug_fn(A, b, pivot)

        for j in range(i + 1, n):
            factor = A[j][i] / A[i][i]
            for k in range(i, n):
                A[j][k] -= factor * A[i][k]
            b[j][0] -= factor * b[i][0]

        debug_fn(A, b, pivot)


def backward_pass(A: matrixType, b: matrixType, pivot=False, eps=1.0e-10):
    n = len(A)
    for j in range(n - 1, -1, -1):
        for i in range(j - 1, -1, -1):
            if abs(A[j][j]) > eps:
                factor = A[i][j] / A[j][j]
            else:
                factor = 0.0
            for k in range(j, n):
                A[i][k] -= factor * A[j][k]
            b[i][0] -= factor * b[j][0]
        debug_fn(A, b, pivot)

    for i in range(n):
        if abs(A[i][i]) > eps:
            b[i][0] /= A[i][i]
            A[i][i] = 1.0
        else:
            b[i][0] = 0.0
    debug_fn(A, b, pivot)


def gauss_factorization(A: matrixType, b: matrixType, pivot=False, eps=1.0e-10):
    """Perform Gauss factorization with partial pivoting on matrix A and vertical vector b"""
    if pivot:
        debug["pivot"] = []
    else:
        debug["nopivot"] = []

    n = len(A)
    A = [[A[i][j] for j in range(n)] for i in range(n)]
    b = [[b[i][j] for j in range(1)] for i in range(n)]

    # forward pass
    forward_pass(A, b, pivot, eps)
    backward_pass(A, b, pivot, eps)

    return b


if __name__ == "__main__":
    total = 0
    total_error = 0
    for i in range(2, 9):
        for k in range(1000):
            A_np = np.random.randint(1, 10, (i, i)).astype(float)
            b_np = np.random.randint(1, 10, (i, 1)).astype(float)
            A = A_np.tolist()
            b = b_np.tolist()

            rank = np.linalg.matrix_rank(A_np)

            try:
                x_np = np.linalg.solve(A_np, b_np)
            except np.linalg.LinAlgError:
                continue

            x_nopivot = gauss_factorization(A, b)
            x_pivot = gauss_factorization(A, b, pivot=True)
            x_nopivot_np = np.array(x_nopivot)
            x_pivot_np = np.array(x_pivot)

            eps = 1.0e-4
            if (not np.allclose(x_np, x_nopivot_np, atol=eps) or not np.allclose(
                x_np, x_pivot_np, atol=eps
            )) and rank == i:
                print("Error:")
                print(f"Rank: {rank}")
                print("A:")
                print_matrix(A)
                print()
                print("b:")
                print_matrix(b)
                print()
                print("x_np:")
                print_matrix(x_np)
                print()
                print("x_nopivot:")
                print_matrix(x_nopivot_np)
                print()
                print("x_pivot:")
                print_matrix(x_pivot_np)
                print()
                print("\n")

                # debug no pivot
                for A, b in debug["nopivot"]:
                    print("A:")
                    print_matrix(A)
                    print()
                    print("b:")
                    print_matrix(b)
                    print()
                assert False
                total_error += 1
            total += 1
    print(f"Total errors: {total_error}/{total}")


# [ 9.00 3.00 7.00 3.00 ]
# [ 3.00 9.00 5.00 2.00 ]
# [ 7.00 1.00 5.00 7.00 ]
# [ 2.00 6.00 4.00 1.00 ]

# [ 2.00 ]
# [ 6.00 ]
# [ 3.00 ]
# [ 6.00 ]

# A = [
#     [9.0, 3.0, 7.0, 3.0],
#     [3.0, 9.0, 5.0, 2.0],
#     [7.0, 1.0, 5.0, 7.0],
#     [2.0, 6.0, 4.0, 1.0],
# ]
# b = [[2.0], [6.0], [3.0], [6.0]]

# x = gauss_factorization(A, b)
# print("x:")
# print_matrix(x)
# print()
# for A, b in debug["nopivot"]:
#     print("A:")
#     print_matrix(A)
#     print()
#     # print("b:")
#     # print_matrix(b)
#     # print()
