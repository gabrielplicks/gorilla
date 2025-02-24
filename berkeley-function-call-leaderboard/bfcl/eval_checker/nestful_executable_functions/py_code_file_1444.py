import random

random.seed(42)


def random_selection(k: int) -> List[Union[int, float]]:

    """Randomly selects a number from the range [-k, k] k times, where k is a positive integer.



    Args:

        k: A positive integer.



    Returns:

        A list of the selected numbers.

    """

    return [random.randint(-k, k) for _ in range(k)]

