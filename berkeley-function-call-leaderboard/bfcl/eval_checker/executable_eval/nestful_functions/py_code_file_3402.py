import random

random.seed(42)
import typing



def make_random_id(num_digits: int = 5) -> str:

    """Generates a random ID of the specified number of digits.



    Args:

        num_digits: The number of digits in the ID. Default is 5.



    Returns:

        A string representing the random ID.

    """

    assert isinstance(num_digits, int), "num_digits must be an integer"

    random_id = ''.join(str(random.randint(0, 9)) for _ in range(num_digits))



    return random_id

