from common import matrixType, print_matrix
import numpy as np

debug = {"pivot": [], "nopivot": []}


def forward_pass(A: matrixType, b: matrixType, pivot=False):
    n = len(A)

    # for i in range(n):
    #     if pivot:
    #         for j in range(i + 1, n):
    #             if abs(A[j][i]) > abs(A[i][i]):
    #                 A[i], A[j] = A[j], A[i]
    #                 b[i], b[j] = b[j], b[i]
    #     for j in range(i + 1, n):
    #         if A[i][i] != 0.0:
    #             factor = A[j][i] / A[i][i]
    #         else:
    #             factor = 0.0
    #         for k in range(i, n):
    #             A[j][k] -= factor * A[i][k]
    #         b[j][0] -= factor * b[i][0]

    #     if pivot:
    #         debug["pivot"].append((A, b))
    #     else:
    #         debug["nopivot"].append((A, b))
    for i in range(n):
        if pivot:
            for j in range(i + 1, n):
                if abs(A[j][i]) > abs(A[i][i]):
                    A[i], A[j] = A[j], A[i]
                    b[i], b[j] = b[j], b[i]
        # check if A[i][i] is zero, if yes, swap with the next non-zero row
        if A[i][i] == 0:
            for j in range(i + 1, n):
                if A[j][i] != 0:
                    A[i], A[j] = A[j], A[i]
                    b[i], b[j] = b[j], b[i]
                    break
        for j in range(i + 1, n):
            factor = A[j][i] / A[i][i]
            for k in range(i, n):
                A[j][k] -= factor * A[i][k]
            b[j][0] -= factor * b[i][0]

        if pivot:
            debug["pivot"].append((A, b))
        else:
            debug["nopivot"].append((A, b))


def backward_pass(A: matrixType, b: matrixType, pivot=False):
    n = len(A)
    for j in range(n - 1, -1, -1):
        for i in range(j - 1, -1, -1):
            if A[j][j] != 0.0:
                factor = A[i][j] / A[j][j]
            else:
                factor = 0.0
            for k in range(j, n):
                A[i][k] -= factor * A[j][k]
            b[i][0] -= factor * b[j][0]
        if pivot:
            debug["pivot"].append((A, b))
        else:
            debug["nopivot"].append((A, b))

    for i in range(n):
        if A[i][i] != 0.0:
            b[i][0] /= A[i][i]
        else:
            b[i][0] = 0.0
    if pivot:
        debug["pivot"].append((A, b))
    else:
        debug["nopivot"].append((A, b))


def gauss_factorization(A: matrixType, b: matrixType, pivot=False):
    """Perform Gauss factorization with partial pivoting on matrix A and vertical vector b"""
    if pivot:
        debug["pivot"] = []
    else:
        debug["nopivot"] = []

    n = len(A)
    A = [[A[i][j] for j in range(n)] for i in range(n)]
    b = [[b[i][j] for j in range(1)] for i in range(n)]

    # forward pass
    forward_pass(A, b, pivot)
    backward_pass(A, b, pivot)

    return b


if __name__ == "__main__":
    total = 0
    total_error = 0
    for i in range(2, 7):
        for k in range(100):
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
