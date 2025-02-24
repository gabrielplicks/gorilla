import random

random.seed(42)
import typing



def group_random_subsets(numbers: typing.List[int], k: int) -> typing.List[typing.List[int]]:

    """Groups a list of numbers into random k-sized subsets.



    Args:

        numbers: The list of numbers to group.

        k: The size of the subsets.



    Returns:

        A list of subsets.

    """

    random.shuffle(numbers)

    return [numbers[i:i+k] for i in range(0, len(numbers), k)]

