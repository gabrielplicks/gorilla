from typing import List



def format_initializer_list(numbers: List[int]) -> str:

    """Formats a list of integers as a C++ initializer list.



    Args:

        numbers: A list of integers.



    Returns:

        A string representation of the formatted list.

    """

    formatted_list = "{" + ", ".join(map(str, numbers)) + "}"

    return formatted_list

