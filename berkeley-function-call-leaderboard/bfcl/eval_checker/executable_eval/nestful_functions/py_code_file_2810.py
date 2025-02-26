from typing import List



def elementwise_division(list_a: List[int], list_b: List[int]) -> List[float]:

    """Calculates the element-wise division of two lists of integers.



    Args:

        list_a: The first list of integers.

        list_b: The second list of integers.



    Returns:

        A list of floats representing the element-wise division of the first list by the second list.

    """

    return [a / b for a, b in zip(list_a, list_b)]

