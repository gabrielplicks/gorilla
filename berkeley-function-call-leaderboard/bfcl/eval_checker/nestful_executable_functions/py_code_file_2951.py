from typing import List



def filter_multiples_of_5(lst: List[int]) -> List[int]:

    """Filters out multiples of 5 from a list of integers.



    Args:

        lst: A list of integers.



    Returns:

        A list of integers that are not multiples of 5.

    """

    return [num for num in lst if num % 5 != 0]

