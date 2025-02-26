from typing import List, Union



def is_magic_sum(numbers: List[Union[int, float]]) -> bool:

    """Checks if the sum of the first two numbers in a list is equal to the last number.



    Args:

        numbers: A list of numbers.



    Returns:

        True if the sum of the first two numbers is equal to the last number, False otherwise.

    """

    if not numbers or len(numbers) < 3:

        return False

    if not all(isinstance(x, (int, float)) for x in numbers):

        return False

    return numbers[0] + numbers[1] == numbers[-1]

