from typing import Dict



def get_val_or_default(d: Dict[str, int], k: str) -> int:

    """Checks if the key `k` is present in the dictionary `d`, and if so, returns its value.

    Otherwise, inserts the key and sets its value to zero, and then returns the value.

    Args:

        d: A dictionary that maps string keys to integer values.

        k: The key to check for in the dictionary.

    """

    if k in d:

        return d[k]

    else:

        d[k] = 0

        return d[k]

