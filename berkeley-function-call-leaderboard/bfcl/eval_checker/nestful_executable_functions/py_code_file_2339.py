from collections import defaultdict

from typing import List, Tuple



def create_dict_from_list_of_pairs(input_list: List[Tuple[str, int]]) -> defaultdict:

    """Creates a dictionary from a list of key-value pairs.

    Each key in the input list becomes a key in the dictionary, and each value becomes a list of values for that key.

    The order of the values in the list for each key is preserved from the input list.

    The values in the dictionary are mutable, meaning you can modify a list value in the dictionary after creation.

    Args:

        input_list: A list of key-value pairs.

    Returns:

        A dictionary where each key is a key from the input list, and each value is a list of values for that key.

    """

    d = defaultdict(list)

    for k, v in input_list:

        d[k].append(v)

    return d

