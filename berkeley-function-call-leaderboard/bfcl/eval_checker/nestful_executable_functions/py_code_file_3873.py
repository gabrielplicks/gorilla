from typing import List



def find_extremes(numbers: List[float]) -> List[float]:

    """Finds the smallest and largest numbers in a given list.



    Args:

        numbers: A list of numbers.



    Returns:

        A two-element list containing the smallest and largest numbers in the list.

        If the list is empty, returns an empty list.

    """

    if not numbers:

        return []

    else:

        return [min(numbers), max(numbers)]

