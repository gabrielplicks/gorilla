from typing import List, Tuple



def get_tensor_values(tensor: List[List[List[int]]], dictionary: dict) -> List[Tuple[Tuple[int, int, int], str, int]]:

    """

    Returns a list of 3-tuples representing the values stored in a 3D tensor.

    Each 3-tuple contains the index, key, and value of a tensor element.

    Args:

        tensor: A 3D tensor of size `n x n x n`.

        dictionary: A dictionary of size `n`.

    """

    values = []

    for i in range(len(tensor)):

        for j in range(len(tensor[0])):

            for k in range(len(tensor[0][0])):

                value = tensor[i][j][k]

                key = dictionary[value]

                index = (i, j, k)

                values.append((index, key, value))

    return values

