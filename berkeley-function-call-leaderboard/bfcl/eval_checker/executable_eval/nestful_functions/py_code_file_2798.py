from random import randint, shuffle

import random
random.seed(42)
from typing import List



def generate_and_sort_random_list() -> List[int]:

    """Generates a list of random integers, shuffles it, and sorts it in ascending order.



    The length of the output list is a random number between 10 and 50.

    The integers in the output list are random numbers between 0 and 99.



    Returns:

        A list of random integers sorted in ascending order.

    """

    length = randint(10, 50)  # Generates a random number between 10 and 50

    random_list = [randint(0, 99) for _ in range(length)]  # Generates a list of random integers

    shuffle(random_list)  # Shuffles the list

    sorted_list = sorted(random_list)  # Sorts the list in ascending order

    return sorted_list

