from typing import Dict, List



def add_keys(d: Dict[str, int], l: List[str]) -> Dict[str, int]:

    """Adds keys from a list to a dictionary and sets their value to 0.



    Args:

        d: The dictionary to add keys to.

        l: The list of keys to add.



    Returns:

        The modified dictionary.

    """

    for key in l:

        d[key] = 0

    return d

