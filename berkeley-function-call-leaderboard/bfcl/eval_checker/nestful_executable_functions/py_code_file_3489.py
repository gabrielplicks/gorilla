from typing import List, Tuple



def max_value_and_index(numbers: List[float]) -> Tuple[float, int]:

    """Finds the maximum value of a list of numbers and its index.



    Args:

        numbers: A list of numbers.



    Returns:

        A tuple containing the maximum value and its index.

    """

    max_value = numbers[0]

    max_index = 0

    for i, num in enumerate(numbers[1:], start=1):

        if num > max_value:

            max_value = num

            max_index = i



    return max_value, max_index

