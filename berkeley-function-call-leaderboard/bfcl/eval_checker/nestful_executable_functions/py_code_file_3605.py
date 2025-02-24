from typing import Dict



def to_dict_sorted(d: Dict[int, str]) -> Dict[int, str]:

    """Converts a dictionary to a dictionary where the keys are sorted in descending order.



    Args:

        d: The dictionary to convert.



    Returns:

        A dictionary with the keys sorted in descending order.

    """

    return {k: d[k] for k in sorted(d, reverse=True)}

