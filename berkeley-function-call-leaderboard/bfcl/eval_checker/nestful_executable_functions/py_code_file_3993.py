from typing import List, Union



def min_max(data: List[Union[int, float]]) -> Union[tuple, None]:

    """Returns the minimum and maximum values of the given data, which is a list of numbers.

    If the input data is invalid, the function returns None.

    Args:

        data: A list of numbers.

    """

    if not isinstance(data, list) or len(data) == 0:  # Precondition check

        return None



    minimum = data[0]

    maximum = data[0]



    for element in data:

        if element < minimum:

            minimum = element

        if element > maximum:

            maximum = element



    return minimum, maximum

