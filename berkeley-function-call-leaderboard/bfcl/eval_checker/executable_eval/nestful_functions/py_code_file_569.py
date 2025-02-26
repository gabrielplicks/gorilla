import math



def compute_square_roots(numbers: list[float]) -> list[float]:

    """Computes the square roots of all positive numbers in the input list.



    Args:

        numbers: A list of numbers.



    Returns:

        A list of square roots of positive numbers in the input list.

    """

    return [math.sqrt(x) for x in numbers if x > 0]

