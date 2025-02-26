from typing import List



def unary_operator(x: List[int]) -> List[int]:

    """Applies a unary operator to each element of the input list and returns the resulting list.



    The unary operator is defined as:

    $$

    f(\mathbf{x}) = 2\mathbf{x} + 1

    $$



    Args:

        x: The input list of integers.



    Returns:

        The resulting list of integers.

    """

    result = []

    for element in x:

        result.append(2 * element + 1)

    return result

