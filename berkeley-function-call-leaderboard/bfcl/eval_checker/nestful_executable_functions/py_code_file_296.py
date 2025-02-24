import random

random.seed(42)
from typing import List



def rand_string(n: int) -> str:

    """Generates a random string of length `n`.



    Args:

        n: The length of the random string.



    Returns:

        A random string of length `n`.

    """

    characters: List[str] = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

    random_characters: List[str] = random.sample(characters, n)

    random_string: str = "".join(random_characters)

    return random_string

