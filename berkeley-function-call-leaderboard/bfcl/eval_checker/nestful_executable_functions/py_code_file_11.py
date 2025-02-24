import random

random.seed(42)
import string



def rand_str(num_chars: int) -> str:

    """Generates a pseudo-random string of `num_chars` characters.

    The string is composed of lowercase letters, digits, and underscores (`_`).

    Args:

        num_chars: The number of characters in the string.

    """

    potential_chars = string.ascii_lowercase + string.digits + "_"

    return ''.join(random.choice(potential_chars) for _ in range(num_chars))

