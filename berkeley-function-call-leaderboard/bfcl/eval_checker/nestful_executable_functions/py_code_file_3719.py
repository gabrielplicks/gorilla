from typing import List, Union



def sum_of_positive(numbers: List[Union[int, float]]) -> Union[int, float, None]:

    """Calculates the sum of all positive numbers in a list.

    Args:

        numbers: A list of numbers.

    Returns:

        The sum of all positive numbers in the list, or None if the list is empty.

    """

    if not isinstance(numbers, list):

        return None

    total = 0

    for num in numbers:

        if num > 0:

            total += num

    return total if numbers else None

