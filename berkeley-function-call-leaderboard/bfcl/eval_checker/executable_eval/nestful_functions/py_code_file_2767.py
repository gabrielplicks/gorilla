import random

random.seed(42)
import string



def random_string(length: int, charset: str = string.ascii_letters + string.digits + string.punctuation) -> str:

    """Generates a random string of given length and with characters from a given set.

    If no set is provided, uses the standard alphabet (uppercase and lowercase letters, digits, and punctuation).

    Args:

        length: The length of the random string to generate.

        charset: The set of characters to use for generating the random string.

    """

    result = []

    for _ in range(length):

        result.append(random.choice(charset))

    return "".join(result)

