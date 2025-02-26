from typing import List



def has_duplicate_items(lst: List[int]) -> bool:

    """Checks if a list has duplicate items without using the `set` function.



    Args:

        lst: The list of numbers to check for duplicates.



    Returns:

        A boolean value indicating whether or not the list has duplicate items.

    """

    for i in range(len(lst)):

        for j in range(i + 1, len(lst)):

            if lst[i] == lst[j]:

                return True

    return False

