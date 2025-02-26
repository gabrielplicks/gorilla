from itertools import permutations

from typing import List, Dict



def get_all_configurations(keys: List[str], values: List[str]) -> List[Dict[str, str]]:

    """

    Returns a list of dictionaries with all possible combinations of key-value pairs.

    The keys are unique in each dictionary, but the same key can appear in multiple dictionaries.

    The values are not unique in any dictionary nor in the returned list.



    Args:

        keys: A list of keys.

        values: A list of values.

    """

    value_permutations = permutations(values)

    configurations = []

    for value_permutation in value_permutations:

        configuration = {}

        for i, key in enumerate(keys):

            configuration[key] = value_permutation[i]

        configurations.append(configuration)

    return configurations

