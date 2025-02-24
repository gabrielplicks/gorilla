from typing import List



def filter_integers(values: List[int]) -> List[int]:

    """Filters a list of integers to only include those that are greater than 10 and divisible by 3.



    Args:

        values: A list of integers.



    Returns:

        A list of integers that are greater than 10 and divisible by 3.

    """

    filtered_values = []



    for value in values:

        if value > 10 and value % 3 == 0:

            filtered_values.append(value)



    return filtered_values

