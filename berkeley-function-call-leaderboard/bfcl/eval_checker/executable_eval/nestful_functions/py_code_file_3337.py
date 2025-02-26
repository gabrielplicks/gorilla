from typing import Dict



def find_dict_max(d: Dict[int, float]) -> float:

    """Finds the maximum value in a dictionary, given its keys are integers.



    Args:

        d: The dictionary to find the maximum value in.



    Returns:

        The maximum value in the dictionary.

    """

    return max(d.values())

