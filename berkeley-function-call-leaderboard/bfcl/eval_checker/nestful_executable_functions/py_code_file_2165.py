from typing import List, Union



def median(data: List[Union[int, float]]) -> Union[float, None]:

    """Calculates the median value of a list of numbers.



    Args:

        data: A list of numbers.



    Returns:

        The median value of the list. If the list is empty, returns None.

    """

    if len(data) == 0:

        return None

    data.sort()

    n = len(data)

    if n % 2 == 1:

        return data[n // 2]

    else:

        return (data[n // 2 - 1] + data[n // 2]) / 2

