import random

random.seed(42)
import typing



def get_random_seed(int_list: typing.List[int]) -> int:

    """Generates a random integer between 0 and 99999 using a seed value from a given list of integers.



    Args:

        int_list: A list of integers. The first integer in the list is used as the seed value.



    Returns:

        A random integer between 0 and 99999.

    """

    random.seed(int_list[0])

    random_seed = random.randint(0, 99999)



    return random_seed

