from typing import List



def check_duplicates(list_one: List[str], list_two: List[str]) -> bool:

    """Checks if two lists have the same elements without duplicates.



    Args:

        list_one: The first list.

        list_two: The second list.



    Returns:

        A boolean value indicating whether the two lists have the same elements without duplicates.

    """

    return set(list_one) == set(list_two)

