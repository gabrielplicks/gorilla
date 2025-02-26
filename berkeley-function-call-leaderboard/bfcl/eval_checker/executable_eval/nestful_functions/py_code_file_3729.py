from typing import List



def max_change(numbers: List[int]) -> int:

    """Calculates the maximum change between any two integers in a list.

    Args:

        numbers: A list of integers.

    Returns:

        The maximum change between any two integers in the list.

    """

    if len(numbers) < 2:

        return 0



    numbers.sort()

    return numbers[-1] - numbers[0]

