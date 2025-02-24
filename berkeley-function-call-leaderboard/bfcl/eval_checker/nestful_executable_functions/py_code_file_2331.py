import random

random.seed(42)


def generate_matrix_of_3d_vectors() -> list[list[float]]:

    """Generates a 3 by 3 matrix of random numbers, where each row is a randomly generated 3D vector.

    The absolute value of each random number is less than or equal to 1.

    Returns:

        A 3 by 3 matrix of random numbers.

    """

    matrix = []

    for _ in range(3):

        row = []

        for _ in range(3):

            value = random.random() * 2 - 1

            row.append(value)

        matrix.append(row)

    return matrix

