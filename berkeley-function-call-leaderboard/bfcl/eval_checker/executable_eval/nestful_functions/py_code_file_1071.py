import random

random.seed(42)
import string



def generate_random_string_of_length(length: int) -> str:

    """Generates a string of random characters with a given length.

    The characters are randomly selected from the ASCII lower-case alphabet (a-z),

    the ASCII upper-case alphabet (A-Z), and the numeric digits (0-9).

    Args:

        length: The length of the string to generate.

    """

    characters = string.ascii_lowercase + string.ascii_uppercase + string.digits

    return ''.join(random.choice(characters) for _ in range(length))

