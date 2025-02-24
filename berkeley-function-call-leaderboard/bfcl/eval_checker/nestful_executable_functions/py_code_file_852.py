from typing import List



def concatenate_numbers(arr: List[int]) -> str:

    """Concatenates the numbers from the list into a string, with a comma separating the numbers.



    Args:

        arr: A list of integers.



    Returns:

        A string that concatenates the numbers from the list, with a comma separating the numbers.

        If the list is empty, returns an empty string.

    """

    if not arr:

        return ''



    return ','.join(str(n) for n in arr)

