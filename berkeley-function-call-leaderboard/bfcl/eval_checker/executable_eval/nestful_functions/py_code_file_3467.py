from typing import List



def perform_operation(numbers: List[int]) -> List[float]:

    """Performs a specific operation on each element of a list of numbers.

    Args:

        numbers: A list of integers.

    Returns:

        A list of floats after performing the operation on each element.

    """

    return list(map(lambda x: 0 if x <= 0 else x*2 if x % 2 else x/2, numbers))

