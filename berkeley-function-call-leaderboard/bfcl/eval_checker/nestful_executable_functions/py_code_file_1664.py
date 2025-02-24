from typing import Dict, List, Any



def subset(d: Dict[Any, Any], i: List[Any]) -> Dict[Any, None]:

    """

    Returns the subset of `d` corresponding to the indicies in `i`.

    Args:

        d: The dictionary to subset.

        i: The list of indices to subset by.

    """

    return dict.fromkeys(i, None)

