from typing import List



def concat_values_from_start(values: List[str], start: int) -> str:

    """Concatenates the elements of a list of strings starting from a given index.



    Args:

        values: A list of strings.

        start: An integer index.



    Returns:

        A string that concatenates the elements of `values` starting from the `start` index.

        If the `start` index is beyond the length of the list, the function returns an empty string.

    """

    if start >= len(values):

        return ""



    concatenated_string = ""

    for i in range(start, len(values)):

        concatenated_string += values[i]



    return concatenated_string

