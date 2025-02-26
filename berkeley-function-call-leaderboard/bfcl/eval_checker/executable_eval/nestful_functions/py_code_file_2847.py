import random

random.seed(42)
import string



def gen_random_str_with_lower_case(n: int) -> str:

    """Generates a random string of length `n` consisting of only lower case letters.



    Args:

        n: The length of the random string.



    Returns:

        A random string of length `n` consisting of only lower case letters.

    """

    lowercase_letters = string.ascii_lowercase

    random_letters = [random.choice(lowercase_letters) for _ in range(n)]

    return ''.join(random_letters)

