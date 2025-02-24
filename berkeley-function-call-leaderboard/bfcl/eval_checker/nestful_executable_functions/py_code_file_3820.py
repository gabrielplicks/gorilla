from typing import List, Tuple



def sort_tuple_by_second(l: List[Tuple[str, str]]) -> List[Tuple[str, str]]:

    """Sorts a list of string tuples `l` by the second element of each tuple in lexicographic order.



    Args:

        l: A list of string tuples.



    Returns:

        The sorted list.

    """

    l.sort(key=lambda x: str.casefold(x[1]))

    return l

