from typing import List



def sort_string_of_ints(input_string: str) -> List[int]:

    """Sorts a string of comma-separated integers in ascending order.



    Args:

        input_string: A string of comma-separated integers.



    Returns:

        A list of integers sorted in ascending order.

    """

    substrings = input_string.split(",")

    integers = [int(substring) for substring in substrings]

    sorted_integers = sorted(integers)

    return sorted_integers

