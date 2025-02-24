import uuid  # Import the uuid module

from typing import List



def generate_unique_keys(strings: List[str]) -> List[str]:

    """Generates a unique key for each string in a list of strings.

    The key is a string converted from a UUID generated for each string.

    The function returns a list of these keys in the same order as the input strings.



    Args:

        strings: A list of strings.



    Returns:

        A list of unique keys.

    """

    keys = []

    for string in strings:

        key = uuid.uuid4()

        key_str = str(key)

        keys.append(key_str)

    return keys

