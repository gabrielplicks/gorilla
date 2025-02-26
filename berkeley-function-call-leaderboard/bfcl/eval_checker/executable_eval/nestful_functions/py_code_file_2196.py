from typing import List



def convert_list_to_string(numbers: List[int]) -> str:

    """Converts a list of numbers into a comma-separated string.



    Args:

        numbers: A list of numbers.



    Returns:

        A string in which each number is separated by a comma.

    """

    return ','.join(map(str, numbers))

