from typing import List



def sort_unique_integers(integers: List[int]) -> List[int]:

    """Sorts a list of integers without duplicates.



    Args:

        integers: A list of integers.



    Returns:

        A list of unique integers in sorted order.

    """

    unique_integers = set(integers)

    sorted_list = sorted(unique_integers)

    return sorted_list

