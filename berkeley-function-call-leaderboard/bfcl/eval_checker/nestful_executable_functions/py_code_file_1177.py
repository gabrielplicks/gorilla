from typing import Dict, Tuple



def sorted_key_value_pairs(dictionary: Dict[str, int]) -> list[Tuple[str, int]]:

    """Returns a sorted list of tuples in the form [('key_1', 'value_1'), ('key_2', 'value_2'), ...] from a dictionary.

    Filters out any key-value pairs whose values are not positive integers.



    Args:

        dictionary: The input dictionary.



    Returns:

        A sorted list of tuples.

    """

    pairs = list(dictionary.items())

    filtered_pairs = [(key, value) for (key, value) in pairs if isinstance(value, int) and value > 0]

    sorted_pairs = sorted(filtered_pairs, key=lambda x: x[0])



    return sorted_pairs

