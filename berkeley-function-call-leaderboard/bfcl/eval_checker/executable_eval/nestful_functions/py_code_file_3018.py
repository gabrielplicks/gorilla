from typing import List



def scale_vectors(vector1: List[int], vector2: List[int]) -> List[int]:

    """Scales each element in `vector1` with the corresponding element in `vector2`.

    If `vector2` is shorter than `vector1`, the function extends `vector2` to the length of `vector1`

    by repeating the last element of `vector2`.

    Args:

        vector1: A list of integers.

        vector2: A list of integers.

    Returns:

        A new list with the scaled values of `vector1`.

    """

    scaled_vector = []

    vector2_length = len(vector2)

    for i, elem in enumerate(vector1):

        if i < vector2_length:

            scaled_elem = elem * vector2[i]

        else:

            scaled_elem = elem * vector2[-1]

        scaled_vector.append(scaled_elem)

    return scaled_vector

