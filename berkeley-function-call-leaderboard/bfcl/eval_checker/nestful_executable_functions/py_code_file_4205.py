import random

random.seed(42)
import typing



def generate_and_sort_list() -> typing.List[int]:

    """Generates a list of 100 random integers between 1 and 1000 (inclusive), and then sorts this list in place.



    Returns:

        A sorted list of 100 random integers between 1 and 1000.

    """

    numbers = [random.randint(1, 1000) for _ in range(100)]

    numbers.sort()

    return numbers

