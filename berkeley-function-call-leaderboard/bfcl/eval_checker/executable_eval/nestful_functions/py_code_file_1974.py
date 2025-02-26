import random

random.seed(42)
import string



def random_alphanumeric_string(length: int) -> str:

    """Generates a random string of a specified length, containing only alphanumeric characters.



    Args:

        length: The length of the random string to generate.



    Returns:

        A random string of the specified length, containing only alphanumeric characters.

    """

    random_characters = []

    for _ in range(length):

        random_characters.append(random.choice(string.ascii_letters + string.digits))

    return ''.join(random_characters)

