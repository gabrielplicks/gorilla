from typing import List



def matrix_scale_multiply(matrix: List[List[float]], scale_value: float) -> List[List[float]]:

    """Multiplies each element in the input matrix by the scale value to create a new matrix.



    Args:

        matrix: The input matrix.

        scale_value: The scale value.

    """

    result = []

    for row in matrix:

        scaled_row = [x * scale_value for x in row]

        result.append(scaled_row)

    return result

