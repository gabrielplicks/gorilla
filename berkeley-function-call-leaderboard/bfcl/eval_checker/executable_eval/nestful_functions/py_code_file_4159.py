from typing import List



def filter_between(data: List[float], threshold: float, upper_bound: float) -> List[float]:

    """

    Filters a list of numbers to only include those within a specified range.



    Args:

        data: A list of numbers.

        threshold: The lower bound of the range.

        upper_bound: The upper bound of the range.

    """

    filtered = []



    for item in data:

        if threshold < item < upper_bound:

            filtered.append(item)



    return filtered

