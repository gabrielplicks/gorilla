from typing import List, Tuple



def filter_truthy_tuples(tuples: List[Tuple[bool, bool, bool]]) -> List[Tuple[bool, bool, bool]]:

    """

    Filters a list of tuples to only include those with at least one truthy value.



    Args:

        tuples: A list of tuples, where each tuple has exactly three boolean elements.



    Returns:

        A list of tuples, where each tuple has at least one truthy value.

    """

    return [(a, b, c) for (a, b, c) in tuples if a or b or c]

