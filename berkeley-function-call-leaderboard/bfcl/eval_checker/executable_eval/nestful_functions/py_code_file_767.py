from typing import List



def set_sym_diff(list_a: List[int], list_b: List[int]) -> List[int]:

    """Calculates the symmetric difference of two lists.

    Args:

        list_a: The first list.

        list_b: The second list.

    """

    set_a = set(list_a)

    set_b = set(list_b)

    return list(set_a ^ set_b)

