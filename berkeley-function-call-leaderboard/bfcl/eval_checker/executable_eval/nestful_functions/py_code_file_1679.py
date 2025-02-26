from typing import List



def convert_to_unique_list(numbers: List[int]) -> List[int]:

    """Converts a list of integers to a list of unique integers.



    Args:

        numbers: A list of integers.



    Returns:

        A list of unique integers.

    """

    return [number for number in set(numbers)]

