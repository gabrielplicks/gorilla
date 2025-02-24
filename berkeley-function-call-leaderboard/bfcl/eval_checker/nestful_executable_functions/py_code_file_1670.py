from typing import List, Tuple



def remove_duplicates_from_list_of_tuples(lst: List[Tuple[int, int]]) -> List[Tuple[int, int]]:

    """Removes duplicates from a list of tuples, while preserving the order of the tuples.



    Args:

        lst: A list of tuples.



    Returns:

        A new list without duplicates.

    """

    unique_tuples = set()

    for tup in lst:

        unique_tuples.add(tup)

    return list(unique_tuples)

