from typing import List, Union



def calc_avg(numbers: List[Union[int, float]]) -> Union[float, None]:

    """Calculates the average of a list of numbers.



    Args:

        numbers: A list of numbers.



    Returns:

        The average of the numbers, or None if the list is empty or contains no numbers.

    """

    total = 0

    count = 0

    for value in numbers:

        if isinstance(value, (int, float)):

            total += value

            count += 1

    if count == 0:

        return None

    return total / count

