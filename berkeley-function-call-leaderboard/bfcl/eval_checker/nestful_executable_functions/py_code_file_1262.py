from typing import List, Union



def find_largest_value(data: List[Union[int, float, str, None]]) -> float:

    """Finds the largest value from a list of input values.



    Args:

        data: A list of input values.



    Returns:

        The largest value from the input list. If a value is None, it is treated as zero.

        The result is rounded to two decimal places.

    """

    largest = 0



    for value in data:

        if value is None:

            continue

        if not isinstance(value, int):

            value = int(value)

        largest = max(largest, value)



    return round(largest, 2)

