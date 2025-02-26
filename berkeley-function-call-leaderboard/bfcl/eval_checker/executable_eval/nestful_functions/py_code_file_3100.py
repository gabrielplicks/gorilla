from typing import List, Tuple



def sort_by_second_element(tuples: List[Tuple[str, int]]) -> List[Tuple[str, int]]:

    """Sorts a list of tuples by their second element in descending order.

    The first element of each tuple is a string, and the second element is an integer.

    Args:

        tuples: A list of tuples to be sorted.

    """

    return sorted(tuples, key=lambda x: x[1], reverse=True)

