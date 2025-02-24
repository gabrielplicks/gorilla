import random

random.seed(42)


def smallest_element(numbers: List[float]) -> float:

    """Randomly shuffles a list of numbers and returns the element with the smallest value.



    Args:

        numbers: A list of numbers.



    Returns:

        The smallest element in the shuffled list.

    """

    random.shuffle(numbers)

    return min(numbers)

