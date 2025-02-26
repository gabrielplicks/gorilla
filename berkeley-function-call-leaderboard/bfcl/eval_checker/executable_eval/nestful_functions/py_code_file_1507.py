from typing import Dict



def filter_dictionary(d: Dict[str, int], v: int) -> Dict[str, int]:

    """

    Filters a dictionary based on a value threshold.



    Args:

        d: The dictionary to filter.

        v: The value threshold.

    """

    return {k: d[k] for k in d if d[k] > v}

