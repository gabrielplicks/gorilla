import random

random.seed(42)
import typing



def random_shuffle(elements: typing.List[typing.Any], seed: int) -> typing.List[typing.Any]:

    """Performs a random shuffle on a list using a given random seed.



    Args:

        elements: The list of elements to be shuffled.

        seed: The random seed to be used for shuffling the list.

    """

    shuffled_list = elements.copy()

    random.seed(seed)

    random.shuffle(shuffled_list)

    return shuffled_list

