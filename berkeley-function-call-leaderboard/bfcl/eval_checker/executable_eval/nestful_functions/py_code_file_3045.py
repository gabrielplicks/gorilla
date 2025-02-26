from typing import Tuple



def find_start_end_indices(input_str: str, sub_str: str) -> Tuple[int, int]:

    """

    Finds the start and end indices of the first occurrence of a sub-string in an input string.



    Args:

        input_str: The input string to search in.

        sub_str: The sub-string to search for.



    Returns:

        A tuple containing the start and end indices of the first occurrence of the sub-string in the input string.

        If the sub-string is not found, returns (-1, -1).

    """

    start_index = input_str.find(sub_str)

    if start_index == -1:

        return (-1, -1)

    end_index = start_index + len(sub_str) - 1

    return (start_index, end_index)

