from typing import Dict



def sum_values(dictionary: Dict[str, float]) -> float:

    """Sum the values of a dictionary, but first validate that all the values in the dictionary are numbers.



    Args:

        dictionary: A dictionary where the values are numbers.



    Raises:

        TypeError: If any of the values in the dictionary are not numbers.

    """

    total = 0

    for value in dictionary.values():

        try:

            total += value

        except TypeError:

            raise TypeError("All values in the dictionary must be numbers.")

    return total

