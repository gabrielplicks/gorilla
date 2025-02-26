from typing import List



def filter_even_index(lst: List[int]) -> List[int]:

    """Creates a new list containing only the elements of the input list that have an even index.



    Args:

        lst: The input list.



    Returns:

        A new list containing only the elements of the input list that have an even index.

    """

    return [el for i, el in enumerate(lst) if i % 2 == 0]

