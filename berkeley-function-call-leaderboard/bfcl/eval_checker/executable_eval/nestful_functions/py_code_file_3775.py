from typing import List, Tuple



def good_data(data: List[Tuple[str, str]]) -> List[Tuple[str, str]]:

    """

    Returns a list of tuples that contain only non-empty strings.



    Args:

        data: A list of tuples.



    Returns:

        A list of tuples that contain only non-empty strings.

    """

    if data is None or len(data) == 0:

        return []



    return [(i, j) for i, j in data if i and j]

