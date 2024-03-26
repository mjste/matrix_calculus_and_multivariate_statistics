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


def gauss_factorization(A: matrixType, b: matrixType, pivot=False, eps=1.0e-8):
    """Perform Gauss factorization with partial pivoting on matrix A and vertical vector b"""
    if pivot:
        debug["pivot"] = []
    else:
        debug["nopivot"] = []

    A = copy_matrix(A)
    b = copy_matrix(b)

    # forward pass
    forward_pass(A, b, pivot, eps)
    backward_pass(A, b, pivot, eps)

    return b


def test_all():
    total = 0
    total_error = 0
    nopivot_fails = 0
    for i in range(2, 12):
        for k in range(5000):
            A_np = np.random.randint(1, 10, (i, i)).astype(float)
            b_np = np.random.randint(1, 10, (i, 1)).astype(float)
            A = A_np.tolist()
            b = b_np.tolist()

            rank = np.linalg.matrix_rank(A_np)

            try:
                x_np = np.linalg.solve(A_np, b_np)
            except np.linalg.LinAlgError:
                continue

            try:
                x_nopivot = gauss_factorization(A, b)
            except ValueError:
                nopivot_fails += 1
                continue
            except ZeroDivisionError:
                nopivot_fails += 1
                continue

            x_pivot = gauss_factorization(A, b, pivot=True)
            x_nopivot_np = np.array(x_nopivot)
            x_pivot_np = np.array(x_pivot)

            eps = 1.0e-4
            if (
                not np.allclose(x_np, x_nopivot_np, atol=eps)
                or not np.allclose(x_np, x_pivot_np, atol=eps)
            ) and rank == i:
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
                # assert False
                total_error += 1
            total += 1
    print(f"Total errors: {total_error}/{total}")
    print(f"Total no pivot fails: {nopivot_fails}")


if __name__ == "__main__":
    # big matrix
    found = False
    while not found:
        dim = 200
        A = np.random.rand(dim, dim).astype(float)
        b = np.random.rand(dim, 1).astype(float)

        try:
            solution = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            continue
        found = True

        A_lst = A.tolist()
        b_lst = b.tolist()

        dirpath = "02/"

        gf_pivot = np.array(gauss_factorization(A_lst, b_lst, pivot=True))
        gf_nopivot = np.array(gauss_factorization(A_lst, b_lst))

        np.savetxt(dirpath + "A.csv", A, delimiter=",")
        np.savetxt(dirpath + "b.csv", b, delimiter=",")
        np.savetxt(dirpath + "gf_pivot.csv", gf_pivot, delimiter=",")
        np.savetxt(dirpath + "gf_nopivot.csv", gf_nopivot, delimiter=",")
        np.savetxt(dirpath + "np_solution.csv", solution, delimiter=",")
