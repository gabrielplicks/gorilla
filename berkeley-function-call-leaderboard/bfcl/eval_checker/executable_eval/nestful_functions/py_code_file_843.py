import random

random.seed(42)


def randomize_dict(d: dict) -> dict:

    """Randomizes the order of the keys in a dictionary while keeping the values the same.



    Args:

        d: The dictionary to be randomized.



    Returns:

        A new dictionary with the same keys and values as the input dictionary,

        but with the keys in a random order.

    """

    keys = d.keys()

    values = d.values()

    new_d = dict.fromkeys(keys, None)

    shuffled_keys = list(keys)

    random.shuffle(shuffled_keys)

    for key, value in zip(shuffled_keys, values):

        new_d[key] = value

    return new_d

