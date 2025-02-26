from typing import Dict



def recursion(d: Dict[str, int]) -> Dict[str, Dict[int, int]]:

    """Recursively replaces each value in the input dictionary with a dictionary.



    The new dictionary has the original value as the key and the number of times

    the value appears in the original dictionary as the value.



    Args:

        d: The input dictionary.



    Returns:

        A new dictionary with the same keys as the input dictionary and the

        values replaced as described above.

    """

    new_d: Dict[str, Dict[int, int]] = {}

    for key, value in d.items():

        if type(value) is int:

            new_d[key] = {value: 1}

        elif type(value) is dict:

            new_d[key] = recursion(value)

    return new_d

