from typing import Tuple



def convert_triplet(triplet_str: str) -> Tuple[str, str, str]:

    """Converts a string containing a structured triplet in the format "key1, key2, key3" to a tuple of three elements where each element is a string representing a key in the triplet.



    Args:

        triplet_str: The string containing the structured triplet.



    Returns:

        A tuple of three elements where each element is a string representing a key in the triplet.

    """

    keys = triplet_str.split(", ")

    return tuple(keys)

