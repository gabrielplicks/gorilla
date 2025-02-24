import random

random.seed(42)
from typing import List



def generate_random_index(input_list: List[int]) -> int:

    """Generates a random index based on the probability distribution of each element's occurrence in the input list.



    Args:

        input_list: A list of integers.



    Returns:

        The random index.

    """

    total_sum = sum(input_list)

    random_num = random.randint(0, total_sum)

    cumulative_sum = 0

    random_index = -1



    for index, elem in enumerate(input_list):

        cumulative_sum += elem

        if cumulative_sum >= random_num:

            random_index = index

            break



    return random_index

