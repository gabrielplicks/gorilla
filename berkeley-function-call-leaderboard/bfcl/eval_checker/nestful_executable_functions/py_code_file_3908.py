from typing import Dict, Any



def find_same_values(a: Dict[str, Any], b: Dict[str, Any]) -> tuple:

    """Finds two different keys in the two dictionaries that have the same value.

    Args:

        a: A dictionary.

        b: Another dictionary.

    Returns:

        A tuple of two keys that have the same value, or None if no such pair exists.

    """

    for key_a in a:

        if a[key_a] in b.values():

            key_b = next(k for k, v in b.items() if v == a[key_a])

            return (key_a, key_b)

    return None

