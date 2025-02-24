from typing import Dict



def create_map_from_args(**kwargs: int) -> Dict[str, int]:

    """Creates a map that associates each argument with a value, where the value is an increasing index starting from 0.

    The function uses the argument name as the key in the map.

    Args:

        kwargs: Arbitrary keyword arguments.

    Returns:

        A dictionary representing the arguments and their values.

    """

    result = {}

    for i, key in enumerate(kwargs):

        result[key] = i

    return result

