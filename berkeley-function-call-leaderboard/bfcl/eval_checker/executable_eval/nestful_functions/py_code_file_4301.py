from typing import List, Tuple



def calc_expected_value(data: List[Tuple[float, float]]) -> float:

    """Calculates the expected value of a function given a list of input-output examples.



    Args:

        data: A list of 2-tuples that each represent an input-output example of a function.



    Returns:

        The expected value of the function.

    """

    total = 0

    for _, output in data:

        total += output

    return total / len(data)

