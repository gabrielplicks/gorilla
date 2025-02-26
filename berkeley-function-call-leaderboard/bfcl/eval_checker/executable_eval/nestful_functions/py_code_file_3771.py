import statistics



def stat(numbers: list[float]) -> tuple[int, float, float, float]:

    """Calculates the number of items, sum, mean, and standard deviation of a list of numbers.



    Args:

        numbers: A list of numbers.



    Returns:

        A tuple containing the number of items, sum, mean, and standard deviation of the numbers.

    """

    length = len(numbers)

    total = sum(numbers)

    mean = total / length

    std = statistics.stdev(numbers)

    return length, total, mean, std

