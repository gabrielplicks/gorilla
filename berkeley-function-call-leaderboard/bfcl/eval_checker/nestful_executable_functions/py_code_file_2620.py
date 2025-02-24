from typing import Tuple



def get_middle_values(values: Tuple[int, int, int]) -> Tuple[int, int]:

    """Returns a tuple of the middle two values from the input tuple.



    Args:

        values: A tuple of three integer values.



    Raises:

        ValueError: If the input tuple does not contain exactly three values.

    """

    if len(values) != 3:

        raise ValueError("Invalid input: Expected a tuple of three values")



    sorted_values = sorted(values)

    return (sorted_values[1], sorted_values[2])

