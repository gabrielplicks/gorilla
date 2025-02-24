from typing import List, Tuple



def sort_tuples_by_value(tuples: List[Tuple[str, int]]) -> List[Tuple[str, int]]:

    """Sorts a list of tuples (name, value) by the value, where name is a string and value is a number.

    If two tuples have the same value, then they should be sorted in descending order by name.

    Returns a new list containing the sorted tuples.



    Args:

        tuples: A list of tuples (name, value) to be sorted.



    Returns:

        A new list containing the sorted tuples.

    """

    return sorted(tuples, key=lambda x: (x[1], -ord(x[0][0])))

