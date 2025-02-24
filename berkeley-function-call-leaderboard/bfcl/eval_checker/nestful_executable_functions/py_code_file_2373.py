from collections import defaultdict

from typing import Dict, List



def group_keys_by_value(d: Dict[int, int]) -> List[List[int]]:

    """Groups the keys in a dictionary by their corresponding values.



    Args:

        d: A dictionary where the keys and values are integers.



    Returns:

        A list of lists, where each inner list contains the keys that have the same corresponding value in the input dictionary.

    """

    grouped = defaultdict(list)

    for key, value in d.items():

        grouped[value].append(key)

    return list(grouped.values())

