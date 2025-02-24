import random

random.seed(42)
import typing



def random_key(dictionary: typing.Dict[str, int]) -> str:

    """Selects a random key from the given dictionary.



    Args:

        dictionary: The dictionary to select a key from.



    Returns:

        A randomly selected key from the dictionary.

    """

    keys = list(dictionary.keys())

    random_index = random.randint(0, len(keys) - 1)

    return keys[random_index]

