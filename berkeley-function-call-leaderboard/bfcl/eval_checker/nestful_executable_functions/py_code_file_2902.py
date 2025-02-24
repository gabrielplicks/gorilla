import random

random.seed(42)
import string



lowercase_letters = string.ascii_lowercase

numbers = string.digits

all_characters = lowercase_letters + numbers



def generate_char() -> str:

    """Generates a random character from a list of lowercase letters and numbers.



    Args:

        None



    Returns:

        A random character from the list of lowercase letters and numbers.

    """

    return random.choice(all_characters)



def generate_string(length: int = 32) -> str:

    """Generates a pseudo-random string of letters and numbers.



    Args:

        length: The length of the string. Default is 32.



    Returns:

        A pseudo-random string of letters and numbers.

    """

    return ''.join([generate_char() for _ in range(length)])

