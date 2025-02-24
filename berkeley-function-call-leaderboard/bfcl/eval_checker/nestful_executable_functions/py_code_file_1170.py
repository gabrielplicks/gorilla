from typing import Dict



def insert_into_dict(d: Dict[str, int], k: str, v: int) -> Dict[str, int]:

    """Inserts a key-value pair into a dictionary in-place.



    Args:

        d: The dictionary to insert the key-value pair into.

        k: The key to be inserted.

        v: The value associated with the key.



    Returns:

        The modified dictionary.

    """

    if d.setdefault(k, v) is not None:

        d[k] = v

    return d

