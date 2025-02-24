from typing import List



def matrix_multiply_2(m: List[List[float]], n: List[List[float]]) -> List[List[float]]:

    """Calculates the product of two matrices.



    Args:

        m: The first matrix.

        n: The second matrix.



    Returns:

        The product of the two matrices.

    """

    result = []

    for i in range(len(m)):

        row = []

        for j in range(len(n[0])):

            dot_product = sum(m[i][k] * n[k][j] for k in range(len(m[0])))

            row.append(dot_product)

        result.append(row)



    return result

