from typing import List



def get_values_only_in_a(a: List[int], b: List[int]) -> List[int]:

    """

    Returns a list containing only the values that are present in `a` but not in `b`.



    Args:

        a: A list of integers.

        b: A list of integers.



    Returns:

        A list of integers.

    """

    values_only_in_a = []

    for item in a:

        if item not in b:

            values_only_in_a.append(item)

    return values_only_in_a

