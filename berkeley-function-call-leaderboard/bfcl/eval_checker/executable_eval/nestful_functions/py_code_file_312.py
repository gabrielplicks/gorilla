from typing import List



def calculate_ab_error(a: List[float], b: List[float]) -> List[float]:

    """Calculates the error between two lists of numbers and returns a list of the errors.



    Args:

        a: A list of numbers.

        b: A list of numbers.



    Raises:

        AssertionError: If the lists a and b do not have the same length.

    """

    assert len(a) == len(b), "The lists a and b must have the same length."

    a_minus_b = [a_i - b_i for a_i, b_i in zip(a, b)]



    return a_minus_b

