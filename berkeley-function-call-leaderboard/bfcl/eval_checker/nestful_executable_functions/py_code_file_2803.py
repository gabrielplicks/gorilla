from typing import List



def remove_all_duplicates(lst: List[int]) -> List[int]:

    """Removes all duplicates from a list and keeps the last occurrence of each element.

    Args:

        lst: The input list.

    """

    indices = {}

    new_lst = []

    for i, elem in enumerate(lst):

        if elem not in indices:

            indices[elem] = i

            new_lst.append(elem)

    return new_lst

