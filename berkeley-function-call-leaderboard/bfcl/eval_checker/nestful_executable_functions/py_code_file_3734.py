from collections import defaultdict

from typing import Dict, List



def group_keys(data: Dict[int, int]) -> Dict[int, List[int]]:

    """Groups the keys into a list for each unique value (sorted from smaller to larger).



    Args:

        data: A dictionary with arbitrary number of keys.



    Returns:

        A new dictionary that groups the keys into a list for each unique value (sorted from smaller to larger).

    """

    result = defaultdict(list)

    for key, value in data.items():

        result[value].append(key)

    for value in result:

        result[value] = sorted(result[value])

    return result

