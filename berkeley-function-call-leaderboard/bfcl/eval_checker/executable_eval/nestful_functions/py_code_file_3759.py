from typing import List



def shorten_list(lst: List[int]) -> List[int]:

    """Removes duplicate values from a list of integers and returns a new list with unique values in reverse order.



    Args:

        lst: A list of integers.



    Returns:

        A new list with unique values in reverse order.

    """

    seen = set()

    shortened_list = []

    for i in reversed(lst):

        if i not in seen:

            seen.add(i)

            shortened_list.append(i)

    return shortened_list

