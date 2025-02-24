from typing import List



def array_difference(array_1: List[int], array_2: List[int]) -> List[int]:

    """Returns a new array that contains the difference between two arrays.

    The return array contains all elements present in `array_1` that are not in `array_2`.

    The function handles cases where elements may appear more than once in either array.

    Args:

        array_1: The first array.

        array_2: The second array.

    """

    return list(set(array_1).difference(set(array_2)))

