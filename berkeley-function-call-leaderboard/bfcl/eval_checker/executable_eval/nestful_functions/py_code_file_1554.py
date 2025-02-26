from typing import List, Tuple

import operator



def get_last_item_in_tuples(l: List[Tuple[int, int, int]]) -> List[int]:

    """Returns a list of the last elements in each tuple of `l`.



    Args:

        l: A list of 3-tuples.



    Returns:

        A list of the last elements in each tuple of `l`.

    """

    p = list(map(operator.itemgetter(-1), l))

    return p

