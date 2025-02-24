from typing import Dict



def num_common_elements(dict1: Dict, dict2: Dict) -> int:

    """Finds the number of elements that two dictionaries have in common.



    Args:

        dict1: The first dictionary.

        dict2: The second dictionary.



    Returns:

        The number of elements that are common between the dictionaries.

    """

    values1 = dict1.values()

    values2 = dict2.values()

    common_values = set(values1).intersection(values2)

    return len(common_values)

