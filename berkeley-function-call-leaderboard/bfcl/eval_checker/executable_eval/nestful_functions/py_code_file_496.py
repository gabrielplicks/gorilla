from typing import List, Tuple



def sum_by_key(tuple_list: List[Tuple[str, int]]) -> dict:

    """Calculates the sum of values for each key in a list of tuples.

    Args:

        tuple_list: A list of tuples with a string key and an integer value.

    """

    key_sum = {}

    for key, value in tuple_list:

        if key in key_sum:

            key_sum[key] += value

        else:

            key_sum[key] = value

    return key_sum

