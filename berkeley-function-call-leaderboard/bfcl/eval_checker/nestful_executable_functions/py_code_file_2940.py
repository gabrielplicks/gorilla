import math



def calculate_std_dev(data: list[float]) -> float:

    """Calculates the sample standard deviation of a given list of numbers.

    Args:

        data: A list of data points.

    """

    mean = sum(data) / len(data)

    squared_differences = [(x - mean) ** 2 for x in data]

    std_dev = math.sqrt(sum(squared_differences) / (len(data) - 1))

    return std_dev

