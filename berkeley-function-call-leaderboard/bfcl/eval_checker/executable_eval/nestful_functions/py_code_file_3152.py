from typing import List, Union



def sum_numeric_elements(elements: List[Union[int, float]]) -> Union[int, float, None]:

    """Calculates the sum of all numeric elements in a list.

    If any element is not numeric, the function returns None.

    Args:

        elements: A list of elements to sum.

    """

    total_sum = 0

    for element in elements:

        if isinstance(element, (int, float)):

            total_sum += element

        else:

            return None

    return total_sum

