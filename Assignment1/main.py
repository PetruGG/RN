import pathlib
import math


def load_system(path: pathlib.Path) -> tuple[list[list[float]], list[float]]:
    A = []
    B = []
    for line in path.read_text().splitlines():
        left, right = line.split('=')
        B.append(float(right.strip()))
        equation = left.strip()
        terms = equation.split()
        coeffs = [0.0, 0.0, 0.0]
        sign = 1
        for term in terms:
            if term == "+":
                sign = 1
            elif term == "-":
                sign = -1
            else:
                if 'x' in term:
                    coeff = term.replace('x', '')
                    if coeff == '':
                        coeffs[0] = sign * 1.0
                    else:
                        coeffs[0] = sign * float(coeff)
                elif 'y' in term:
                    coeff = term.replace('y', '')
                    if coeff == '':
                        coeffs[1] = sign * 1.0
                    else:
                        coeffs[1] = sign * float(coeff)
                elif 'z' in term:
                    coeff = term.replace('z', '')
                    if coeff == '':
                        coeffs[2] = sign * 1.0
                    else:
                        coeffs[2] = sign * float(coeff)
                sign = 1
        A.append(coeffs)
    return A, B


def determinant(matrix: list[list[float]]) -> float:
    n = len(matrix)
    if n == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    else:
        a11, a12, a13 = matrix[0]
        a21, a22, a23 = matrix[1]
        a31, a32, a33 = matrix[2]
        return a11 * (a22 * a33 - a23 * a32) - a12 * (a21 * a33 - a23 * a31) + a13 * (a21 * a32 - a22 * a31)


def trace(matrix: list[list[float]]) -> float:
    n = len(matrix)
    sum = 0.0
    for i in range(n):
        sum += matrix[i][i]
    return sum


def norm(vector: list[float]) -> float:
    sum = 0.0
    for element in vector:
        sum += element ** 2
    return math.sqrt(sum)


def transpose(matrix: list[list[float]]) -> list[list[float]]:
    n = len(matrix)
    transposed = []
    for j in range(n):
        new_row = []
        for i in range(n):
            new_row.append(matrix[i][j])
        transposed.append(new_row)
    return transposed


def multiply(matrix: list[list[float]], vector: list[float]) -> list[float]:
    n = len(matrix)
    result = [0.0] * n
    for i in range(n):
        for j in range(n):
            result[i] += matrix[i][j] * vector[j]
    return result


def copy_matrix(matrix: list[list[float]]) -> list[list[float]]:
    return [row[:] for row in matrix]


def solve_cramer(matrix: list[list[float]], vector: list[float]) -> list[float]:
    solution = []
    Ax = copy_matrix(matrix)
    Ay = copy_matrix(matrix)
    Az = copy_matrix(matrix)
    det_matrix = determinant(matrix)
    for i in range(3):
        Ax[i][0] = vector[i]
        Ay[i][1] = vector[i]
        Az[i][2] = vector[i]

    x = determinant(Ax) / det_matrix
    y = determinant(Ay) / det_matrix
    z = determinant(Az) / det_matrix
    solution.append(x)
    solution.append(y)
    solution.append(z)
    return solution


def minor(matrix: list[list[float]], i: int, j: int) -> list[list[float]]:
    new_matrix = []
    for k, row in enumerate(matrix):
        if k != i:
            new_row = []
            for p, element in enumerate(row):
                if p != j:
                    new_row.append(element)
            new_matrix.append(new_row)
    return new_matrix


def cofactor(matrix: list[list[float]]) -> list[list[float]]:
    n = len(matrix)
    new_matrix = []
    for i in range(n):
        new_row = []
        for j in range(n):
            minor_matrix = minor(matrix, i, j)
            cofactor_value = ((-1) ** (i + j)) * determinant(minor_matrix)
            new_row.append(cofactor_value)
        new_matrix.append(new_row)
    return new_matrix


def adjoint(matrix: list[list[float]]) -> list[list[float]]:
    return transpose(cofactor(matrix))


def solve(matrix: list[list[float]], vector: list[float]) -> list[float]:
    inv_matrix = []
    adj_matrix = adjoint(matrix)
    det_a = determinant(matrix)
    for i in range(len(adj_matrix)):
        inv_row = []
        for j in range(len(adj_matrix)):
            inv_row.append(adj_matrix[i][j] / det_a)
        inv_matrix.append(inv_row)
    return multiply(inv_matrix, vector)


if __name__ == '__main__':
    A, B = load_system(pathlib.Path("system.txt"))
    print(f"{A=} {B=}")
    print(f"{determinant(A)=}")
    print(f"{trace(A)=}")
    print(f"{norm(B)=}")
    print(f"{transpose(A)=}")
    print(f"{multiply(A, B)=}")
    print(f"{solve_cramer(A, B)=}")
    print(f"{solve(A, B)=}")
