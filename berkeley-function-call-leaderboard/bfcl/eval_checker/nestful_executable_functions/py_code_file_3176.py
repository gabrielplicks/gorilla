import random

random.seed(42)
from typing import Dict



def random_hash(length: int) -> str:

    """Generates a random string of size `length`, where each character is randomly chosen from the set of digits (0-9).



    Args:

        length: The length of the random string.



    Returns:

        A random string of length `length`.

    """

    return ''.join(random.choices('0123456789', k=length))



def setup_hash_fixtures() -> Dict[str, str]:

    """Generates 10 random hashes of length 8 and accessible by an identifier `test_hash_1` to `test_hash_10`.



    Returns:

        A dictionary containing the 10 random hashes.

    """

    fixtures = {}

    for i in range(1, 11):

        fixtures[f'test_hash_{i}'] = random_hash(8)

    return fixtures

