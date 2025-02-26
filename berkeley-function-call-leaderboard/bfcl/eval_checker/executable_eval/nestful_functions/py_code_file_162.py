from typing import Callable

import random

random.seed(42)


def hash_fn(str_input: str) -> int:

    """A mock hash function that returns a random integer between 1 and 2^32 (inclusive).

    Args:

        str_input: The string to be hashed.

    """

    return random.randint(1, 2**32)



def hash_str_and_return_id(str_input: str, hash_fn: Callable[[str], int]) -> tuple:

    """Returns a tuple of the provided string and its hashed value using the provided hash function.

    Args:

        str_input: The string to be hashed.

        hash_fn: The hash function to be used.

    """

    hash_value = hash_fn(str_input)

    return (str_input, hash_value)

