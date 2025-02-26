import random

random.seed(42)
import string

from typing import List



def generate_key(length: int) -> str:

    """Generates a random key of the specified length.



    Args:

        length: The length of the key.



    Returns:

        A random key of the specified length.

    """

    characters = string.ascii_lowercase + string.digits

    key = ''

    for _ in range(length):

        key += random.choice(characters)

    return key



def generate_keys(num_keys: int, length: int) -> List[str]:

    """Generates a list of unique keys of the specified length.



    Args:

        num_keys: The number of keys to generate.

        length: The length of each key.



    Returns:

        A list of unique keys of the specified length.

    """

    keys = set()

    while len(keys) < num_keys:

        keys.add(generate_key(length))

    return list(keys)

