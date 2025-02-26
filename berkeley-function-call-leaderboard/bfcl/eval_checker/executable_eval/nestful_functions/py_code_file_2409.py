from typing import List



def scale_to_zero_mean_and_unit_variance(x: List[float]) -> List[float]:

    """Scales a list of numbers to have zero mean and unit variance.



    Args:

        x: The list of numbers to scale.



    Raises:

        ValueError: If the input is not a list.

    """

    if not isinstance(x, list):

        raise ValueError("Input must be a list")

    try:

        mu = sum(x) / len(x)

        sigma = (sum([(xi - mu)**2 for xi in x]) / len(x))**(1/2)

    except ZeroDivisionError:  # If the list is empty

        return []

    return [(xi - mu) / sigma for xi in x]

