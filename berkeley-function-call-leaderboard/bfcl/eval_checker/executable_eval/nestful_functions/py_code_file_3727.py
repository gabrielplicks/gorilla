from typing import List, Tuple

from operator import itemgetter



def sort_tuples_by_age(tuples: List[Tuple[str, int]]) -> List[Tuple[str, int]]:

    """Sorts a list of tuples based on the age.



    Args:

        tuples: A list of tuples containing name and age.



    Returns:

        A sorted list of tuples based on the age.

    """

    return sorted(tuples, key=itemgetter(1))

