from typing import List



def max_percentage_sum(args: List[float]) -> float:

    """Calculates the percentage of the provided parameters' maximum value.



    Args:

        args: A list of float parameters.



    Returns:

        The percentage of the provided parameters' maximum value.

    """

    if len(args) == 0:

        return 0

    if len(args) == 1:

        return args[0]



    max_value = max(args)

    sum_value = sum(args)

    percentage = (sum_value / max_value) * 100



    return percentage

