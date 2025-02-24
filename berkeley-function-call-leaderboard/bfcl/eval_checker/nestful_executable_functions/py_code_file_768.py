from typing import List, Tuple, Set



class TupleComparison(tuple):

    def __eq__(self, other):

        return tuple(self) == tuple(other)



    def __hash__(self):

        return hash(tuple(self))



def remove_duplicates_from_tuple_list(tup_list: List[TupleComparison]) -> List[TupleComparison]:

    """

    Removes duplicate tuples from a list of tuples and returns a new list containing

    only unique tuples. This function utilizes tuple comparison and list comprehension.



    Args:

        tup_list: A list of tuples.



    Returns:

        A new list containing only unique tuples.

    """

    unique_tuples: Set[TupleComparison] = set(tup_list)

    return [tuple(tup) for tup in unique_tuples]

