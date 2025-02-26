from typing import List



def greater_than_or_equal_to(list_of_ints: List[int], num: int) -> List[int]:

    """

    Returns a list of integers that are greater than or equal to the given integer.



    Args:

        list_of_ints: A list of integers.

        num: An integer.



    Returns:

        A list of integers that are greater than or equal to the given integer.

    """

    return [n for n in list_of_ints if n >= num]

